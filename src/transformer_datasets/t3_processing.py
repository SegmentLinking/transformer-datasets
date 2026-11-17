import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait, FIRST_COMPLETED
from argparse import ArgumentParser

import uproot
import awkward as ak
import numpy as np

import torch
from torch_geometric.data import Data

import hashlib

from transformer_datasets.constants import SIM_VARS, T3_VARS, T3_HIT_SIMIDX, T3_HIT_LAYER, T3_PMATCHED, LS_INDEX, LS_VARS, MD_VARS, MD_INDEX
from transformer_datasets.constants import TARGET, FAKE_TARGET, LAYER_INFO, MD_SIMIDX, ALL_COLUMNS, MIN_T3_PT, MAX_T3_PT, UNMATCHED
from transformer_datasets.constants import N_T3_MDS, MD_HIT_INDEX


class GraphBuilder:
    def __init__(self, input_paths, output_path, train_split=0.8):
        self.output_path = output_path
        self.train_split = train_split
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        self.train_path = os.path.join(self.output_path, "train")
        self.val_path = os.path.join(self.output_path, "val")
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)

        if isinstance(input_paths, str):
            input_paths = [input_paths]
        self.input_trees = [uproot.open(p)["tree"] for p in input_paths]

    @staticmethod
    def _to_tensor(values, dtype=torch.float32):
        return torch.as_tensor(np.asarray(values), dtype=dtype)

    @staticmethod
    def _extract_graph_index(filename):
        if not (filename.startswith("graph_") and filename.endswith(".pt")):
            return None

        idx_str = filename[len("graph_"):-len(".pt")]
        if not idx_str.isdigit():
            return None
        return int(idx_str)

    def _load_existing_graph_indices(self):
        existing = set()

        for directory in (self.train_path, self.val_path):
            if not os.path.isdir(directory):
                continue
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    idx = self._extract_graph_index(entry.name)
                    if idx is not None:
                        existing.add(idx)

        return existing

    def _iter_selected_events(self, tree, num_events, selected_local_indices, chunk_size):
        """Yield selected events while reading ROOT arrays in chunks."""
        if not selected_local_indices:
            return

        selected = sorted(selected_local_indices)
        selected_ptr = 0
        n_selected = len(selected)

        for start in range(0, num_events, chunk_size):
            if selected_ptr >= n_selected:
                break

            stop = min(start + chunk_size, num_events)
            if selected[selected_ptr] >= stop:
                continue

            batch = tree.arrays(ALL_COLUMNS, entry_start=start, entry_stop=stop)
            while selected_ptr < n_selected and selected[selected_ptr] < stop:
                local_idx = selected[selected_ptr]
                offset = local_idx - start
                yield local_idx, batch[offset: offset + 1]
                selected_ptr += 1

    def process_event(self, event_data, idx, args):
        """Processes a single event and saves the graph."""
        try:
            if split_value(idx, args.seed) < self.train_split:
                output_file = os.path.join(self.train_path, f"graph_{idx}.pt")
            else:
                output_file = os.path.join(self.val_path, f"graph_{idx}.pt")

            if args.debug:
                ...
            elif not args.overwrite:
                train_exists = os.path.exists(os.path.join(self.train_path, f"graph_{idx}.pt"))
                val_exists = os.path.exists(os.path.join(self.val_path, f"graph_{idx}.pt"))
                if not (train_exists or val_exists):
                    pass
                else:
                    print(f"Graph {idx} already exists, skipping...")
                    return

            t3_df = ak.to_dataframe(event_data[T3_VARS])
            pt_mask = (t3_df["t3_pt"] < MAX_T3_PT) & (t3_df["t3_pt"] > MIN_T3_PT)

            t3_features = t3_df[pt_mask].values

            ls_idx_all = ak.to_dataframe(event_data[LS_INDEX]).values
            md_idx_all = unique_t3_md_indices(ak.to_dataframe(event_data[MD_INDEX]).values[ls_idx_all])
            ls_idx = ls_idx_all[pt_mask]
            md_idx_flat = md_idx_all[pt_mask]

            ls_features = ak.to_dataframe(event_data[LS_VARS]).values[ls_idx]
            ls_features = ls_features.reshape(-1, 2 * len(LS_VARS))

            md_features = ak.to_dataframe(event_data[MD_VARS]).values[md_idx_flat]
            md_features = md_features.reshape(-1, N_T3_MDS * len(MD_VARS))

            md_layer = ak.to_dataframe(event_data[LAYER_INFO]).values[md_idx_flat]
            md_layer = md_layer.reshape(-1, N_T3_MDS)

            md_simIdx = ak.to_dataframe(event_data[MD_SIMIDX]).values[md_idx_flat]
            md_simIdx = md_simIdx.reshape(-1, N_T3_MDS)

            hit_simIdx = convert_hit_simIdx_to_array(event_data[T3_HIT_SIMIDX])
            pMatched_by_hand = calculate_pMatched(hit_simIdx)
            pMatched_ntuple = ak.to_dataframe(event_data[T3_PMATCHED]).values.flatten()
            n_mismatch = np.sum(~np.isclose(pMatched_by_hand, pMatched_ntuple, atol=1e-4))
            if n_mismatch > 0:
                raise ValueError(f"Mismatch between calculated pMatched and ntuple pMatched for graph {idx}")
            hit_simIdx = hit_simIdx[pt_mask]
            hit_layer_all = ak.to_dataframe(event_data[T3_HIT_LAYER]).values
            hit_layer = hit_layer_all[pt_mask]

            hit_idx_all = t3_hit_indices(md_idx_all, ak.to_dataframe(event_data[MD_HIT_INDEX]).values)
            hit_simIdxs_all = hit_sim_indices(event_data[T3_HIT_SIMIDX])
            check_hits_consistent(hit_idx=hit_idx_all, hit_simIdxs=hit_simIdxs_all, layer=hit_layer_all)
            hit_idx = hit_idx_all[pt_mask]
            hit_simIdxs = [slot[pt_mask] for slot in hit_simIdxs_all]

            target_np = ak.to_dataframe(event_data[TARGET])[pt_mask].values

            if args.nofakes:
                fake_mask = ak.to_dataframe(event_data[FAKE_TARGET])[pt_mask].values.flatten() == 0
                node_features_np = np.concatenate([t3_features[fake_mask], ls_features[fake_mask], md_features[fake_mask]], axis=1)
                target_flat = self._to_tensor(target_np[fake_mask]).flatten()
                md_layer = self._to_tensor(md_layer[fake_mask])
                md_simIdx = self._to_tensor(md_simIdx[fake_mask])
                hit_simIdx = self._to_tensor(hit_simIdx[fake_mask])
                hit_layer = self._to_tensor(hit_layer[fake_mask])
                hit_idx = self._to_tensor(hit_idx[fake_mask], dtype=torch.long)
                hit_simIdxs = [self._to_tensor(slot[fake_mask], dtype=torch.long) for slot in hit_simIdxs]

            else:
                node_features_np = np.concatenate([t3_features, ls_features, md_features], axis=1)
                target_flat = self._to_tensor(target_np).flatten()
                target_flat[target_flat < 0] = -999999
                md_layer = self._to_tensor(md_layer)
                md_simIdx = self._to_tensor(md_simIdx)
                hit_simIdx = self._to_tensor(hit_simIdx)
                hit_layer = self._to_tensor(hit_layer)
                hit_idx = self._to_tensor(hit_idx, dtype=torch.long)
                hit_simIdxs = [self._to_tensor(slot, dtype=torch.long) for slot in hit_simIdxs]

            node_features = self._to_tensor(node_features_np)
            sim_features = self._to_tensor(ak.to_dataframe(event_data[SIM_VARS]).values)

            graph = Data(x=node_features,
                         sim_index=target_flat, sim_features=sim_features,
                         md_layer=md_layer, md_simIdx=md_simIdx,
                         hit_layer=hit_layer, hit_simIdx=hit_simIdx, hit_idx=hit_idx,
                         **{f"hit_{k}_simIdxs": slot for k, slot in enumerate(hit_simIdxs)})
            if args.debug:
                print(graph)
                return

            torch.save(graph, output_file)
            print(f"Processed graph {idx}")

        except Exception as e:
            print(f"Error processing graph {idx}: {e}")

    def process_events_in_parallel(self, args):
        n_cpu = os.cpu_count() or 1
        n_workers = min(args.n_workers, max(1, n_cpu // 2)) if not args.debug else 1
        chunk_size = 1 if args.debug else max(1, args.chunk_size)
        max_pending = max(1, n_workers * 4)
        global_idx = 0
        skipped_existing = 0

        existing_indices = set()
        if not args.overwrite:
            existing_indices = self._load_existing_graph_indices()
            if existing_indices:
                print(f"Found {len(existing_indices)} existing graphs. Pre-skipping before ROOT I/O.")

        # Parallelism comes from the process pool, so each worker runs torch single-threaded.
        # Otherwise every worker starts one thread per core, which on many-core nodes hits the
        # thread limit ("libgomp: Thread creation failed") and breaks the pool.
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=torch.set_num_threads,
            initargs=(1,),
        ) as executor:
            pending = set()
            for tree in self.input_trees:
                tree_entries = int(tree.num_entries)
                num_events = tree_entries if args.n_events == -1 else min(args.n_events, tree_entries)
                if args.debug:
                    num_events = 1

                tree_global_start = global_idx
                selected_local_indices = []
                for local_idx in range(num_events):
                    abs_idx = tree_global_start + local_idx
                    if args.overwrite or abs_idx not in existing_indices:
                        selected_local_indices.append(local_idx)
                    else:
                        skipped_existing += 1

                for local_idx, event_data in self._iter_selected_events(tree, num_events, selected_local_indices, chunk_size):
                    abs_idx = tree_global_start + local_idx
                    future = executor.submit(self.process_event, event_data, abs_idx, args)
                    pending.add(future)

                    if len(pending) >= max_pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for finished in done:
                            finished.result()

                    if args.debug:
                        break

                if args.debug:
                    break

                global_idx += num_events

            if pending:
                done, _ = wait(pending)
                for finished in done:
                    finished.result()

        if skipped_existing:
            print(f"Skipped {skipped_existing} existing graph indices before read/submit.")


def split_value(idx: int, seed: int) -> float:
    """
    Deterministic number in [0, 1) for the train/val split of graph idx.

    It depends only on (seed, idx), so an event goes to the same split for any number of
    workers and any processing order. Events with split_value < train_split go to train.
    """
    digest = hashlib.blake2b(f"{seed}:{idx}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64


def unique_t3_md_indices(md_idx: np.ndarray) -> np.ndarray:
    """
    Reduce the MD indices of each T3's two LSs to its three distinct MDs, inner to outer.

    Input: integer array of shape (n_t3, 2, 2), indexed [t3, ls, md in ls].
    Output: integer array of shape (n_t3, N_T3_MDS) = (LS0.md0, LS0.md1, LS1.md1).

    The two LSs of a T3 share their middle MD (LS0.md1 is LS1.md0), so that slot is dropped.
    """
    if not np.array_equal(md_idx[:, 0, 1], md_idx[:, 1, 0]):
        raise ValueError("Found a T3 whose two LSs do not share their middle MD")
    return np.stack([md_idx[:, 0, 0], md_idx[:, 0, 1], md_idx[:, 1, 1]], axis=1)


def t3_hit_indices(md_idx: np.ndarray, md_hit_idx: np.ndarray) -> np.ndarray:
    """
    Ntuple hit indices of the six hits of each T3, in the order of the t3_hit_{0..5}_* branches.

    Inputs: md_idx (n_t3, N_T3_MDS) from unique_t3_md_indices, and md_hit_idx (n_md, 2) with columns
    MD_HIT_INDEX (anchor, other).
    Output: integer array of shape (n_t3, 2 * N_T3_MDS) = (MD0 anchor, MD0 other, MD1 anchor, ...).

    Pixel MDs have anchor == other, so a T3 can list the same hit index twice.
    """
    return md_hit_idx[md_idx].reshape(md_idx.shape[0], 2 * N_T3_MDS)


def hit_sim_indices(sim_lists: ak.Array) -> list[np.ndarray]:
    """
    List the sim tracks of every hit of every T3, one padded array per hit slot.

    Input: one event, with fields T3_HIT_SIMIDX, each of shape (n_t3, var).
    Output: a list of 2 * N_T3_MDS int64 arrays, one per hit slot (hit_0_simIdxs, ...), each of
    shape (n_t3, K) with K the largest number of sim tracks of a hit in that slot (at least 1).

    The candidates of a hit are its distinct non-negative sim indices, largest first, padded with UNMATCHED.
    This is what LST's matchedSimTrkIdxsAndFracsFromCounts counts: a hit counts towards every sim track in its row.
    The ntuple branches have one entry per sim hit, so they contain repeats and negatives.
    """
    columns = [sim_lists[branch][0] for branch in T3_HIT_SIMIDX]  # [0] removes the event dimension
    return [distinct_descending(col[col >= 0]) for col in columns]


def distinct_descending(lists: ak.Array) -> np.ndarray:
    """
    Pad jagged lists of integers into one array: distinct values of each row, largest first, padded with UNMATCHED.

    Input: shape (n, var). Output: int64 array of shape (n, K), K the largest number of distinct values in a row
    (at least 1). Example: [[3, 7, 3], [], [5]] -> [[7, 3], [UNMATCHED, UNMATCHED], [5, UNMATCHED]].
    """
    max_len = int(ak.max(ak.num(lists), initial=1))
    padded = ak.to_numpy(ak.fill_none(ak.pad_none(lists, max(1, max_len), clip=True), UNMATCHED)).astype(np.int64)
    padded = -np.sort(-padded, axis=1)
    repeated = np.zeros_like(padded, dtype=bool)
    repeated[:, 1:] = padded[:, 1:] == padded[:, :-1]
    padded[repeated] = UNMATCHED
    padded = -np.sort(-padded, axis=1)
    width = max(1, int((padded != UNMATCHED).sum(axis=1).max(initial=0)))
    return padded[:, :width]


def check_hits_consistent(
    hit_idx: np.ndarray,
    hit_simIdxs: list[np.ndarray],
    layer: np.ndarray,
) -> None:
    """
    Check that every slot with the same ntuple hit index has the same sim tracks and layer.

    Inputs, for one event: hit_idx (n_t3, 6) from t3_hit_indices, hit_simIdxs from hit_sim_indices,
    and layer (n_t3, 6) from the t3_hit_*_layer branches.

    Raises ValueError otherwise, which would mean the T3 -> LS -> MD -> hit indices do not line up with
    the t3_hit_* branches.
    """
    width = max(slot.shape[1] for slot in hit_simIdxs)
    padded = [np.pad(slot, ((0, 0), (0, width - slot.shape[1])), constant_values=UNMATCHED) for slot in hit_simIdxs]
    sims = np.stack(padded, axis=1).reshape(-1, width)  # (n_t3 * 6, width), row-major (t3, slot)
    _, first_slot, hit = np.unique(hit_idx.ravel(), return_index=True, return_inverse=True)
    if not np.array_equal(sims[first_slot][hit], sims):
        raise ValueError("Slots with the same hit index have different sim tracks")
    if not np.array_equal(layer.ravel()[first_slot][hit], layer.ravel()):
        raise ValueError("Slots with the same hit index have different layers")


def convert_hit_simIdx_to_array(t3_hit_simIdx: ak.Array) -> np.ndarray:
    """
    Collapse the per-hit lists of sim indices into one sim index per hit.

    Input: one event, with fields T3_HIT_SIMIDX, each of shape (n_t3, var).
    Output: integer array of shape (n_t3, 6). Unmatched hits get UNMATCHED.

    Within a T3, each hit takes whichever of its sim indices is shared by the most hits of
    that T3, which maximizes the matched fraction of the T3 (the logic of t3_pMatched).

    Example for the six hits of one T3:

        input:  [[7], [7], [3, 7], [3], [], [7, 3]]
        output: [ 7,   7,   7,      3, -999, 7    ]

    Sim index 7 appears in four hits and sim index 3 in three, so the two hits with a
    choice both take 7. The hit with only [3] keeps 3, and the empty hit is unmatched.
    Taking the first sim index of each hit would instead give [7, 7, 3, 3, -999, 7],
    which lowers the matched fraction of the T3 from 4/6 to 3/6.
    """
    n_events = len(t3_hit_simIdx)
    if n_events != 1:
        raise ValueError(f"Expected exactly 1 event, got {n_events}")

    columns = [t3_hit_simIdx[branch][0] for branch in T3_HIT_SIMIDX] # [0] removes the event dimension

    # Fast path: take the first sim index of every hit. This is already the answer
    # for any T3 where no hit has more than one sim index.
    result = np.stack([ak.to_numpy(ak.fill_none(ak.firsts(col), UNMATCHED)) for col in columns], axis=1)

    # Slow path: only T3s where at least one hit has a choice to make
    has_choice = np.any(np.stack([ak.to_numpy(ak.num(col)) > 1 for col in columns], axis=1), axis=1)
    rows = np.flatnonzero(has_choice)
    choices = [ak.to_list(col[rows]) for col in columns]
    for i_row, row in enumerate(rows):
        hits = [choices[it][i_row] for it in range(len(columns))]
        result[row] = resolve_t3(hits)

    return result


def resolve_t3(hits: list) -> list:
    # Count how many hits of this T3 each sim index appears in
    n_hits = {}
    for sim_idxs in hits:
        for sim_idx in dict.fromkeys(sim_idxs):  # distinct, in order
            n_hits[sim_idx] = n_hits.get(sim_idx, 0) + 1
    # Rank sim indices once per T3, so that every hit breaks ties the same way:
    # most hits first, then first appearance (sorted is stable)
    ranking = sorted(n_hits, key=lambda sim_idx: -n_hits[sim_idx])
    rank = {sim_idx: i for i, sim_idx in enumerate(ranking)}
    return [min(sim_idxs, key=rank.get) if sim_idxs else UNMATCHED for sim_idxs in hits]


def calculate_pMatched(t3_hit_simIdx: np.ndarray) -> np.ndarray:
    """
    Recalculate t3_pMatched from the output of convert_hit_simIdx_to_array.

    Input: integer array of shape (n_t3, n_hits). Output: float array of shape (n_t3,).

    Mirrors matchedSimTrkIdxsAndFracs in trkCore.cc: find the most common sim index among the
    hits (ties: whichever appears first), where unmatched counts as its own category.
    If unmatched is the most common, pMatched is 0.
    """
    n_t3, n_hits = t3_hit_simIdx.shape
    # counts[i, j]: number of hits in T3 i which share the sim index of hit j
    same = t3_hit_simIdx[:, :, None] == t3_hit_simIdx[:, None, :]
    counts = same.sum(axis=2)
    best_hit = np.argmax(counts, axis=1)  # argmax returns the first maximum
    rows = np.arange(n_t3)
    best_simIdx = t3_hit_simIdx[rows, best_hit]
    best_count = counts[rows, best_hit]
    return np.where(best_simIdx == UNMATCHED, 0.0, best_count / n_hits)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input", type=str, nargs="+", required=True)
    argparser.add_argument("--output", type=str, default="./")
    argparser.add_argument("--n_workers", "-n", type=int, default=16)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--overwrite", action="store_true", help="Overwrite existing graphs")
    argparser.add_argument("--n_events", type=int, default=-1, help="Maximum number of events to process (-1 for all)")
    argparser.add_argument("--split", type=float, default=0.8, help="Train/val split ratio")
    argparser.add_argument("--nofakes", action="store_true", help="Exclude fake hits from the graphs")
    argparser.add_argument("--chunk_size", type=int, default=256, help="Number of events to read per ROOT I/O chunk")
    argparser.add_argument("--seed", type=int, default=0,
                           help="Seed of the train/val split (deterministic per graph index; see split_value)")
    args = argparser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    training_data = GraphBuilder(args.input, args.output, args.split)
    training_data.process_events_in_parallel(args)
