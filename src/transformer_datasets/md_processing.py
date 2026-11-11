"""
Build one MD-level dataset file per event from an LSTNtuple.

Each output file is a plain dict of tensors, saved with torch.save and readable with
torch.load(path, weights_only=True):

    x:            float32 (n_md, 10): [layer, rt, cosphi, sinphi, z, drt, dphi, dz, eta, qoverpt]
    sim_index:    int64   (n_md,):   md_simIdx, UNMATCHED for fake MDs
    sim_features: float32 (n_md, 10): MD_SIM_VARS of the matched sim track, UNMATCHED for fake MDs
    hit_index:    int64   (n_md, 2):  LST hit index of the [inner, outer] hit, as in t5_hitIndices
    hit_sim_pairs: int64  (n_pairs, 2): unique [hit index, sim index] pairs, one per sim of each hit of the kept
                  MDs, with no match-fraction cut. Hits with no sim have no pair.
    source_path:  str: absolute path of the LSTNtuple the event was read from
    source_entry: int: entry of the event in that LSTNtuple

<output>/manifest.txt lists every planned event, one line each, as key=value:
    event=event_0.pt set=train source_path=/abs/path/LSTNtuple_X.root source_entry=0

rt, phi and z are those of the inner hit. dX is X(outer hit) - X(inner hit).
eta and qoverpt are LST's own MD estimates: eta is md_eta (anchor hit), and qoverpt in GeV^-1 is the MD q/pT
estimate from md_dphichange, as in md_pt = rt * K2RINV_1GEV / sin(dphichange) but inverted, so it is finite at
dphichange = 0, and with the sign flipped to match sim q.
The inner hit is the one with smaller rt in the barrel and smaller |z| in the endcap.
Pixel LS MDs (md_isPLS) are discarded.
md_simIdx (and so sim_index) is set only when both hits come from one sim. Use hit_index and hit_sim_pairs for
hit-level matching. Needs an LSTNtuple with the md_{anchor,other}_{hitIdx,simIdxs} branches.
"""

import os
import random
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor

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
# Jagged: a list of sims per MD
MD_HIT_SIMS = ["md_anchor_simIdxs", "md_other_simIdxs"]
MD_SIM_VARS = SIM_VARS + ["sim_event", "sim_bunchCrossing"]
COLUMNS = MD_COLUMNS + MD_HIT_SIMS + MD_SIM_VARS

BARREL_LAYERS = range(1, 7)
ENDCAP_LAYERS = range(7, 12)
# Curvature constant (speed of light * B / 2) in GeV^-1 cm^-1, k2Rinv1GeVf in LSTCore interface/alpaka/Common.h
K2RINV_1GEV = 2.99792458e-3 * 3.8 / 2


def main() -> None:
    args = parse_args()
    events = plan_events(
        input_paths=args.input,
        output=args.output,
        n_events=args.n_events,
        train=args.train,
        val=args.val,
        tune=args.tune,
        test=args.test,
        seed=args.seed,
    )
    write_manifest(events=events, output=args.output)
    run_jobs(jobs=events, n_workers=args.n_workers)


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
    parser.add_argument("--train", type=float, default=0.6, help="Training fraction")
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Validation fraction (training loss, early stopping)",
    )
    parser.add_argument(
        "--tune", type=float, default=0.05, help="Tuning fraction (harness thresholds)"
    )
    parser.add_argument(
        "--test", type=float, default=0.2, help="Test fraction (reported numbers)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for the sets")
    return parser.parse_args()


def plan_events(
    input_paths: list[str],
    output: str,
    n_events: int,
    train: float,
    val: float,
    tune: float,
    test: float,
    seed: int | None,
) -> list[tuple[str, int, str]]:
    """
    One (input path, entry, output file) per event, numbered event_0, event_1, ... across all inputs.

    train, val, tune and test are the fractions of events in each set (see assign_sets). Creates
    <output>/{train,val,tune,test}, and raises FileExistsError if one exists: a production always starts from
    scratch (no resuming), so the manifest and the files always come from the same run.
    """
    sources = []
    for input_path in input_paths:
        with uproot.open(input_path) as f:
            n_entries = f["tree"].num_entries
        if n_events != -1:
            n_entries = min(n_entries, n_events)
        sources.extend((input_path, entry) for entry in range(n_entries))

    sets = assign_sets(
        n_events=len(sources),
        train=train,
        val=val,
        tune=tune,
        test=test,
        seed=seed,
    )
    os.makedirs(os.path.join(output, "train"))
    os.makedirs(os.path.join(output, "val"))
    os.makedirs(os.path.join(output, "tune"))
    os.makedirs(os.path.join(output, "test"))
    return [
        (input_path, entry, os.path.join(output, name, f"event_{idx}.pt"))
        for idx, ((input_path, entry), name) in enumerate(zip(sources, sets))
    ]


def assign_sets(
    n_events: int,
    train: float,
    val: float,
    tune: float,
    test: float,
    seed: int | None,
) -> list[str]:
    """
    The set ("train", "val", "tune" or "test") of each event, shape (n_events,).

    Exact sizes: round(fraction * n_events) events for val, tune and test, and the rest for train. Events are
    shuffled with seed before the sets are cut, so the same seed and n_events always give the same sets.
    """
    if abs(train + val + tune + test - 1) > 1e-9:
        raise ValueError(
            f"set fractions must sum to 1, got train={train} val={val} tune={tune} test={test}"
        )
    n_val = round(val * n_events)
    n_tune = round(tune * n_events)
    n_test = round(test * n_events)
    n_train = n_events - n_val - n_tune - n_test
    labels = (
        ["train"] * n_train + ["val"] * n_val + ["tune"] * n_tune + ["test"] * n_test
    )
    order = list(range(n_events))
    random.Random(seed).shuffle(order)
    sets = [""] * n_events
    for event, label in zip(order, labels):
        sets[event] = label
    return sets


def write_manifest(
    events: list[tuple[str, int, str]],
    output: str,
) -> None:
    """Write <output>/manifest.txt: one key=value line per event (see the module docstring)."""
    path = os.path.join(output, "manifest.txt")
    with open(path, "w") as f:
        for input_path, entry, output_file in events:
            event = os.path.basename(output_file)
            split = os.path.basename(os.path.dirname(output_file))
            source_path = os.path.abspath(input_path)
            f.write(
                f"event={event} set={split} source_path={source_path} source_entry={entry}\n"
            )
    print(f"[done] output={path} n_events={len(events)}")


def run_jobs(
    jobs: list[tuple[str, int, str]],
    n_workers: int,
) -> None:
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                process_event,
                input_path=input_path,
                entry=entry,
                output_file=output_file,
            )
            for input_path, entry, output_file in jobs
        ]
        for future in futures:
            future.result()


def process_event(
    input_path: str,
    entry: int,
    output_file: str,
) -> None:
    with uproot.open(input_path) as f:
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
    x = md_features(inner=inner, outer=outer, layer=layer)
    estimates = md_track_estimates(
        eta=md["md_eta"][keep],
        dphichange=md["md_dphichange"][keep],
        anchor=anchor,
    )
    x = np.concatenate([x, estimates], axis=1)

    anchor_inner = anchor_is_inner(anchor=anchor, other=other, layer=layer)
    hit_index = np.where(
        anchor_inner[:, None],
        np.stack([anchor_hit, other_hit], axis=1),
        np.stack([other_hit, anchor_hit], axis=1),
    )
    pairs = hit_sim_pairs(
        anchor_hit=anchor_hit,
        other_hit=other_hit,
        anchor_sims=event["md_anchor_simIdxs"][keep],
        other_sims=event["md_other_simIdxs"][keep],
    )

    sim = np.stack([ak.to_numpy(event[col]) for col in MD_SIM_VARS], axis=1)
    sim_index = np.where(md_simIdx >= 0, md_simIdx, UNMATCHED)

    data = {
        "x": torch.as_tensor(x, dtype=torch.float32),
        "sim_index": torch.as_tensor(sim_index, dtype=torch.int64),
        "sim_features": torch.as_tensor(
            md_sim_features(md_simIdx=md_simIdx, sim_features=sim),
            dtype=torch.float32,
        ),
        "hit_index": torch.as_tensor(hit_index, dtype=torch.int64),
        "hit_sim_pairs": torch.as_tensor(pairs, dtype=torch.int64),
        "source_path": os.path.abspath(input_path),
        "source_entry": entry,
    }
    torch.save(data, output_file)
    print(f"[done] output={output_file} n_md={len(x)} n_hit_sim_pairs={len(pairs)}")


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


def md_track_estimates(
    eta: np.ndarray,
    dphichange: np.ndarray,
    anchor: np.ndarray,
) -> np.ndarray:
    """
    LST's per-MD track estimates [eta, qoverpt], shape (n_md, 2).

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
    result = np.full(
        (len(md_simIdx), sim_features.shape[1]),
        UNMATCHED,
        dtype=np.float32,
    )
    matched = md_simIdx >= 0
    result[matched] = sim_features[md_simIdx[matched]]
    return result


def hit_sim_pairs(
    anchor_hit: np.ndarray,
    other_hit: np.ndarray,
    anchor_sims: ak.Array,
    other_sims: ak.Array,
) -> np.ndarray:
    """
    Unique [hit index, sim index] pairs of all hits of the given MDs, shape (n_pairs, 2), sorted.

    Input: anchor_hit and other_hit of shape (n_md,), and anchor_sims and other_sims with a list of sims per MD.
    A hit shared by several MDs gives its pairs once. A hit with no sim gives no pair.
    """
    hits = np.concatenate(
        [
            np.repeat(anchor_hit, ak.to_numpy(ak.num(anchor_sims))),
            np.repeat(other_hit, ak.to_numpy(ak.num(other_sims))),
        ]
    )
    sims = np.concatenate(
        [
            ak.to_numpy(ak.flatten(anchor_sims)),
            ak.to_numpy(ak.flatten(other_sims)),
        ]
    )
    pairs = np.stack([hits, sims], axis=1).astype(np.int64)
    return np.unique(pairs, axis=0)


if __name__ == "__main__":
    main()
