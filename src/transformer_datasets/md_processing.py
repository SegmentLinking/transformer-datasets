"""
Build one MD-level dataset file per event from an LSTNtuple.

Each output file is a plain dict of tensors, saved with torch.save and readable with
torch.load(path, weights_only=True):

    md_features:      float32 (n_md, 10): [layer, rt, cosphi, sinphi, z, drt, dphi, dz, eta, qoverpt]
    md_simIdx:        int64   (n_md,):   md_simIdx, UNMATCHED for fake MDs
    md_simFeatures:   float32 (n_md, 10): ALL_SIM_VARS of the matched sim track, UNMATCHED for fake MDs
    md_hitIdx:        int64   (n_md, 2):  LST hit index of the [inner, outer] hit, as in t5_hitIndices
    md_inner_simIdxs: int32   (n_md, n_max_simIdxs): sims of the inner hit, padded with UNMATCHED.
    md_outer_simIdxs: int32.  (n_md, n_max_simIdxs): the same for the outer hit
    sim_features:     float32 (n_hs, 10): ALL_SIM_VARS of hard-scatter sims, same columns as md_simFeatures
    lstntuple_file:  str: file name (no directory) of the LSTNtuple the event was read from
    lstntuple_entry: int: entry of the event in that LSTNtuple

Each LSTNtuple file goes whole into one dataset (train, val or test).
An event is written as <output>/<dataset>/<LSTNtuple name>.entry_<N>.pt, e.g. train/LSTNtuple_XXXX.entry_7.pt.

<output>/train.txt, val.txt and test.txt list each dataset's LSTNtuple files, one file name per line, e.g.
    LSTNtuple_X.root

The lists record the LSTNtuple's file name, not its directory: the same input exists as an all-sim LSTNtuple
(read here, for training) and a nominal one (read by the evaluation harness), with the same name and event order.

The inner hit is the one with smaller rt in the barrel and smaller |z| in the endcap.
Pixel LS MDs (md_isPLS) are discarded.
Use md_hitIdx and md_{inner,outer}_simIdxs for hit-level matching.
Needs an LSTNtuple with the md_{anchor,other}_{hitIdx,simIdxs} branches.
"""

import os
import random
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import awkward as ak
import numpy as np
import torch
import uproot

from transformer_datasets.constants import SIM_VARS, UNMATCHED

MD_ANCHOR = ["md_anchor_x", "md_anchor_y", "md_anchor_z"]
MD_OTHER = ["md_other_x", "md_other_y", "md_other_z"]
MD_COLUMNS = (
    MD_ANCHOR
    + MD_OTHER
    + ["md_layer", "md_isPLS", "md_simIdx", "md_anchor_hitIdx", "md_other_hitIdx"]
    + ["md_eta", "md_dphichange"]
)
MD_HIT_SIMS = ["md_anchor_simIdxs", "md_other_simIdxs"]
ALL_SIM_VARS = SIM_VARS + ["sim_event", "sim_bunchCrossing"]
COLUMNS = MD_COLUMNS + MD_HIT_SIMS + ALL_SIM_VARS

# Curvature constant (speed of light * B / 2)
K2RINV_1GEV = 2.99792458e-3 * 3.8 / 2
DATASETS = ["train", "val", "test"]
BARREL_LAYERS = range(1, 7)
ENDCAP_LAYERS = range(7, 12)


@dataclass
class InputFile:
    """One input LSTNtuple: its path, the dataset all its events go to, and how many entries are processed."""

    lstntuple_path: str
    dataset: str
    n_entries: int


@dataclass
class EventJob:
    """One event to process: where to read it, and the .pt file to write."""

    lstntuple_path: str
    lstntuple_entry: int
    output_file: str


def main() -> None:
    args = parse_args()
    files = plan_files(
        input_paths=args.input,
        n_events=args.n_events,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
    )
    jobs = plan_jobs(files=files, output=args.output)
    write_file_lists(files=files, output=args.output)
    run_jobs(jobs=jobs, n_workers=args.n_workers)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--input", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="./")
    parser.add_argument("--n_workers", "-n", type=int, default=8)
    parser.add_argument(
        "--n_events",
        type=int,
        default=-1,
        help="Events to process per input file (-1=all)",
    )
    parser.add_argument(
        "--train", type=float, default=0.7, help="Training fraction of files"
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Validation fraction of files (training loss, early stopping)",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.15,
        help="Test fraction of files (reported numbers)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for the datasets")
    return parser.parse_args()


def plan_files(
    input_paths: list[str],
    n_events: int,
    train: float,
    val: float,
    test: float,
    seed: int | None,
) -> list[InputFile]:
    """
    Every input file with its dataset and number of entries to process, sorted by file name.

    Sorting first makes the datasets depend only on the seed and the set of files, not on the order of --input.
    train, val and test are fractions of files (see assign_datasets). n_events caps the entries per file (-1: all).
    """
    input_paths = sorted(input_paths, key=os.path.basename)
    datasets = assign_datasets(
        n_files=len(input_paths),
        train=train,
        val=val,
        test=test,
        seed=seed,
    )
    files = []
    for input_path, dataset in zip(input_paths, datasets):
        with uproot.open(input_path) as root_file:
            n_entries = root_file["tree"].num_entries
        if n_events != -1:
            n_entries = min(n_entries, n_events)
        files.append(
            InputFile(lstntuple_path=input_path, dataset=dataset, n_entries=n_entries)
        )
    return files


def assign_datasets(
    n_files: int,
    train: float,
    val: float,
    test: float,
    seed: int | None,
) -> list[str]:
    """
    The dataset ("train", "val" or "test") of each file, shape (n_files,).

    Exact sizes: round(fraction * n_files) files for val and test, and the rest for train. Files are shuffled with
    seed before the datasets are cut, so the same seed and n_files always give the same datasets.
    """
    if abs(train + val + test - 1) > 1e-9:
        raise ValueError(
            f"dataset fractions must sum to 1, got train={train} val={val} test={test}"
        )
    n_val = round(val * n_files)
    n_test = round(test * n_files)
    n_train = n_files - n_val - n_test
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    order = list(range(n_files))
    random.Random(seed).shuffle(order)
    datasets = [""] * n_files
    for i_file, label in zip(order, labels):
        datasets[i_file] = label
    return datasets


def plan_jobs(
    files: list[InputFile],
    output: str,
) -> list[EventJob]:
    """
    One job per event, written as <output>/<dataset>/<LSTNtuple name>.entry_<N>.pt.

    Creates <output>/{train,val,test}, and raises FileExistsError if one exists: a production always starts from
    scratch (no resuming), so the file lists and the files always come from the same run.
    """
    for dataset in DATASETS:
        os.makedirs(os.path.join(output, dataset))
    jobs = []
    for file in files:
        basename = os.path.basename(file.lstntuple_path)
        for entry in range(file.n_entries):
            output_file = os.path.join(output, file.dataset, f"{basename}.entry_{entry}.pt")
            jobs.append(
                EventJob(
                    lstntuple_path=file.lstntuple_path,
                    lstntuple_entry=entry,
                    output_file=output_file,
                )
            )
    return jobs


def write_file_lists(
    files: list[InputFile],
    output: str,
) -> None:
    """Write <output>/<dataset>.txt for each dataset: its LSTNtuple file names, one per line."""
    for dataset in DATASETS:
        path = os.path.join(output, f"{dataset}.txt")
        fnames = [
            os.path.basename(file.lstntuple_path)
            for file in files
            if file.dataset == dataset
        ]
        with open(path, "w") as list_file:
            for fname in fnames:
                list_file.write(f"{fname}\n")
        print(f"[done] output={path} n_files={len(fnames)}")


def run_jobs(
    jobs: list[EventJob],
    n_workers: int,
) -> None:
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                process_event,
                input_path=job.lstntuple_path,
                entry=job.lstntuple_entry,
                output_file=job.output_file,
            )
            for job in jobs
        ]
        try:
            for future in futures:
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def process_event(
    input_path: str,
    entry: int,
    output_file: str,
) -> None:
    with uproot.open(input_path) as f:
        print(f"[io] input={input_path} entry={entry}")
        event = f["tree"].arrays(COLUMNS, entry_start=entry, entry_stop=entry + 1)[0]
    md = {col: ak.to_numpy(event[col]) for col in MD_COLUMNS}

    keep = ~md["md_isPLS"]
    anchor = np.stack([md[col] for col in MD_ANCHOR], axis=1)[keep]
    other = np.stack([md[col] for col in MD_OTHER], axis=1)[keep]
    layer = md["md_layer"][keep]
    md_simIdx = md["md_simIdx"][keep]
    anchor_hit = md["md_anchor_hitIdx"][keep]
    other_hit = md["md_other_hitIdx"][keep]

    inner, outer = order_hits(anchor=anchor, other=other, layer=layer)
    md_feats = md_features(inner=inner, outer=outer, layer=layer)
    extra_features = md_extra_features(
        eta=md["md_eta"][keep],
        dphichange=md["md_dphichange"][keep],
        anchor=anchor,
    )
    md_feats = np.concatenate([md_feats, extra_features], axis=1)

    anchor_inner = anchor_is_inner(anchor=anchor, other=other, layer=layer)
    md_hitIdx = np.where(
        anchor_inner[:, None],
        np.stack([anchor_hit, other_hit], axis=1),
        np.stack([other_hit, anchor_hit], axis=1),
    )
    md_inner_simIdxs, md_outer_simIdxs = hit_simIdxs(
        anchor_simIdxs=event["md_anchor_simIdxs"][keep],
        other_simIdxs=event["md_other_simIdxs"][keep],
        anchor_inner=anchor_inner,
    )
    md_simIdx = np.where(md_simIdx >= 0, md_simIdx, UNMATCHED)

    sim_features = np.stack([ak.to_numpy(event[col]) for col in ALL_SIM_VARS], axis=1)
    sim_hard_scatter = (
        (ak.to_numpy(event["sim_event"]) == 0) &
        (ak.to_numpy(event["sim_bunchCrossing"]) == 0)
    )

    data = {
        "md_features": torch.as_tensor(md_feats, dtype=torch.float32),
        "md_simIdx": torch.as_tensor(md_simIdx, dtype=torch.int64),
        "md_simFeatures": torch.as_tensor(
            md_sim_features(md_simIdx=md_simIdx, sim_features=sim_features),
            dtype=torch.float32,
        ),
        "md_hitIdx": torch.as_tensor(md_hitIdx, dtype=torch.int64),
        "md_inner_simIdxs": torch.as_tensor(md_inner_simIdxs, dtype=torch.int32),
        "md_outer_simIdxs": torch.as_tensor(md_outer_simIdxs, dtype=torch.int32),
        "sim_features": torch.as_tensor(sim_features[sim_hard_scatter], dtype=torch.float32),
        "lstntuple_file": os.path.basename(input_path),
        "lstntuple_entry": entry,
    }
    torch.save(data, output_file)
    print(f"[done] output={output_file} n_md={len(md_feats)}")


def anchor_is_inner(
    anchor: np.ndarray,
    other: np.ndarray,
    layer: np.ndarray,
) -> np.ndarray:
    """
    True for MDs whose anchor hit is the inner hit, shape (n_md,).

    Input: anchor and other of shape (n_md, 3) with columns x, y, z, and layer of shape (n_md,).

    Barrel (BARREL_LAYERS): the inner hit has the smaller rt.
    Endcap (ENDCAP_LAYERS): the inner hit has the smaller |z|.
    On a tie the anchor counts as inner.
    Raises ValueError if a layer is in neither BARREL_LAYERS nor ENDCAP_LAYERS.
    """
    is_barrel = np.isin(layer, BARREL_LAYERS)
    is_endcap = np.isin(layer, ENDCAP_LAYERS)
    if not np.all(is_barrel | is_endcap):
        unknown = np.unique(layer[~(is_barrel | is_endcap)])
        raise ValueError(f"Layers in neither barrel nor endcap: {unknown}")
    anchor_rt = np.hypot(anchor[:, 0], anchor[:, 1])
    other_rt = np.hypot(other[:, 0], other[:, 1])
    return np.where(
        is_barrel,
        anchor_rt <= other_rt,
        np.abs(anchor[:, 2]) <= np.abs(other[:, 2]),
    )


def order_hits(
    anchor: np.ndarray,
    other: np.ndarray,
    layer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort the two hits of each MD into (inner, outer), see anchor_is_inner.

    Input: anchor and other of shape (n_md, 3) with columns x, y, z, and layer of shape (n_md,).
    Output: inner and outer of shape (n_md, 3).
    """
    anchor_inner = anchor_is_inner(anchor=anchor, other=other, layer=layer)
    inner = np.where(anchor_inner[:, None], anchor, other)
    outer = np.where(anchor_inner[:, None], other, anchor)
    return inner, outer


def md_features(
    inner: np.ndarray,
    outer: np.ndarray,
    layer: np.ndarray,
) -> np.ndarray:
    """
    Per-MD features [layer, rt, cosphi, sinphi, z, drt, dphi, dz], shape (n_md, 8).

    dphi is wrapped into [-pi, pi).
    """
    inner_rt = np.hypot(inner[:, 0], inner[:, 1])
    outer_rt = np.hypot(outer[:, 0], outer[:, 1])
    inner_phi = np.arctan2(inner[:, 1], inner[:, 0])
    outer_phi = np.arctan2(outer[:, 1], outer[:, 0])
    dphi = (outer_phi - inner_phi + np.pi) % (2 * np.pi) - np.pi
    return np.stack(
        [
            layer,
            inner_rt,
            np.cos(inner_phi),
            np.sin(inner_phi),
            inner[:, 2],
            outer_rt - inner_rt,
            dphi,
            outer[:, 2] - inner[:, 2],
        ],
        axis=1,
    )


def md_extra_features(
    eta: np.ndarray,
    dphichange: np.ndarray,
    anchor: np.ndarray,
) -> np.ndarray:
    """
    Extra MD features: [eta, qoverpt], shape (n_md, 2).

    Input: md_eta and md_dphichange of shape (n_md,), and anchor of shape (n_md, 3) with columns x, y, z.
    qoverpt = -sin(dphichange) / (K2RINV_1GEV * anchor rt), in GeV^-1. The minus sign matches sim q: LST's
    dphichange has the opposite sign convention.
    """
    anchor_rt = np.hypot(anchor[:, 0], anchor[:, 1])
    qoverpt = -np.sin(dphichange) / (K2RINV_1GEV * anchor_rt)
    return np.stack([eta, qoverpt], axis=1)


def md_sim_features(
    md_simIdx: np.ndarray,
    sim_features: np.ndarray,
) -> np.ndarray:
    """
    Sim track features for each MD, shape (n_md, n_sim_vars).

    Rows of MDs with md_simIdx < 0 are filled with UNMATCHED.
    """
    n_md, n_sim_vars = len(md_simIdx), sim_features.shape[1]
    result = np.full((n_md, n_sim_vars), UNMATCHED, dtype=np.float32)
    matched = md_simIdx >= 0
    result[matched] = sim_features[md_simIdx[matched]]
    return result


def hit_simIdxs(
    anchor_simIdxs: ak.Array,
    other_simIdxs: ak.Array,
    anchor_inner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The sims of each MD's inner and outer hit, each shape (n_md, n_max_simIdxs), padded with UNMATCHED.

    Input: anchor_simIdxs and other_simIdxs with a list of sims per MD (md_anchor_simIdxs, md_other_simIdxs), and
    anchor_inner of shape (n_md,) from anchor_is_inner.
    n_max_simIdxs is the most sims of any hit in the event, so both outputs have the same width. Each row keeps
    LST's order of sims.
    """
    # int(): ak.pad_none needs a Python int, not a numpy one
    n_max_simIdxs = int(
        max(
            ak.max(ak.num(anchor_simIdxs), initial=0),
            ak.max(ak.num(other_simIdxs), initial=0),
        )
    )
    # Pad each list to n_max_simIdxs: [[5], []] -> [[5, UNMATCHED], [UNMATCHED, UNMATCHED]]
    anchor = ak.to_numpy(
        ak.fill_none(ak.pad_none(anchor_simIdxs, n_max_simIdxs, clip=True), UNMATCHED)
    )
    other = ak.to_numpy(
        ak.fill_none(ak.pad_none(other_simIdxs, n_max_simIdxs, clip=True), UNMATCHED)
    )
    inner = np.where(anchor_inner[:, None], anchor, other)
    outer = np.where(anchor_inner[:, None], other, anchor)
    return inner, outer


if __name__ == "__main__":
    main()
