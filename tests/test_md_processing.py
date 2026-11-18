import os

import awkward as ak
import numpy as np
import pytest

from transformer_datasets.constants import UNMATCHED
from transformer_datasets.md_processing import (
    K2RINV_1GEV,
    InputFile,
    anchor_is_inner,
    assign_datasets,
    hit_simIdxs,
    md_features,
    md_sim_features,
    md_extra_features,
    order_hits,
    plan_jobs,
    write_file_lists,
)


def test_anchor_is_inner():
    # barrel MD with the anchor outside, endcap MD with the anchor inside
    anchor = np.array([[30.2, 0.0, 1.0], [50.0, 0.0, 150.0]])
    other = np.array([[30.0, 0.0, 5.0], [40.0, 0.0, 150.4]])
    anchor_inner = anchor_is_inner(anchor=anchor, other=other, layer=np.array([2, 7]))
    assert np.array_equal(anchor_inner, [False, True])


def test_anchor_is_inner_tie_keeps_anchor_inner():
    hit = np.array([[30.0, 0.0, 1.0]])
    assert anchor_is_inner(anchor=hit, other=hit, layer=np.array([3]))[0]


def test_anchor_is_inner_rejects_unknown_layer():
    hit = np.array([[30.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        anchor_is_inner(anchor=hit, other=hit, layer=np.array([0]))


def test_hit_simIdxs():
    # MD 0: anchor inner; anchor hit with sim 5, other hit with no sim
    # MD 1: other inner; anchor hit with sim 5, other hit with sims 7 and 5 (LST's order is kept)
    inner, outer = hit_simIdxs(
        anchor_simIdxs=ak.Array([[5], [5]]),
        other_simIdxs=ak.Array([[], [7, 5]]),
        anchor_inner=np.array([True, False]),
    )
    # Width 2: the most sims of any hit
    assert np.array_equal(inner, [[5, UNMATCHED], [7, 5]])
    assert np.array_equal(outer, [[UNMATCHED, UNMATCHED], [5, UNMATCHED]])


def test_hit_simIdxs_no_sims():
    # No hit has a sim: width 0
    inner, outer = hit_simIdxs(
        anchor_simIdxs=ak.Array([[], []]),
        other_simIdxs=ak.Array([[], []]),
        anchor_inner=np.array([True, False]),
    )
    assert inner.shape == (2, 0)
    assert outer.shape == (2, 0)


def test_order_hits_barrel_uses_rt():
    # anchor has the larger rt but the smaller 3D R and |z|: rt decides in the barrel
    anchor = np.array([[30.2, 0.0, 1.0]])
    other = np.array([[30.0, 0.0, 5.0]])
    inner, outer = order_hits(anchor=anchor, other=other, layer=np.array([2]))
    assert np.array_equal(inner, other)
    assert np.array_equal(outer, anchor)


def test_order_hits_endcap_uses_abs_z():
    # negative endcap: anchor has the smaller z but the larger |z|
    anchor = np.array([[50.0, 0.0, -150.4]])
    other = np.array([[60.0, 0.0, -150.0]])
    inner, outer = order_hits(anchor=anchor, other=other, layer=np.array([8]))
    assert np.array_equal(inner, other)
    assert np.array_equal(outer, anchor)


def test_order_hits_no_swap():
    anchor = np.array([[30.0, 0.0, 1.0], [50.0, 0.0, 150.0]])
    other = np.array([[30.2, 0.0, 1.0], [40.0, 0.0, 150.4]])
    inner, outer = order_hits(anchor=anchor, other=other, layer=np.array([5, 7]))
    assert np.array_equal(inner, anchor)
    assert np.array_equal(outer, other)


def test_md_features_values():
    inner = np.array([[0.0, 30.0, 2.0]])
    outer = np.array([[0.0, 31.0, 2.5]])
    x = md_features(inner=inner, outer=outer, layer=np.array([4]))
    assert np.allclose(x, [[4, 30.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.5]])


def test_md_features_dphi_wraps_across_pi():
    # inner just below +pi, outer just above -pi: dphi is small and positive
    inner = np.array([[-30.0, 0.01, 0.0]])
    outer = np.array([[-31.0, -0.01, 0.0]])
    x = md_features(inner=inner, outer=outer, layer=np.array([4]))
    dphi = x[0, 6]
    assert 0 < dphi < 0.01


def test_md_sim_features_unmatched_rows():
    sim = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    result = md_sim_features(md_simIdx=np.array([2, -1, 0]), sim_features=sim)
    assert np.array_equal(result, [[3.0, 30.0], [UNMATCHED, UNMATCHED], [1.0, 10.0]])


def test_md_extra_features():
    # anchor at rt = 50 cm; a positive 2 GeV track has q/pT = 0.5, i.e. dphichange = -asin(K * 50 * 0.5) in LST's
    # convention; a straight MD (dphichange = 0) gives q/pT = 0
    anchor = np.array([[30.0, 40.0, 10.0], [30.0, 40.0, 10.0]])
    dphichange = np.array([-np.arcsin(K2RINV_1GEV * 50.0 * 0.5), 0.0])
    extra_features = md_extra_features(
        eta=np.array([0.2, -1.5]),
        dphichange=dphichange,
        anchor=anchor,
    )
    assert np.allclose(extra_features, [[0.2, 0.5], [-1.5, 0.0]])


FILES = [
    InputFile(lstntuple_path="/data/allsim/a.root", dataset="train", n_entries=2),
    InputFile(lstntuple_path="/data/allsim/b.root", dataset="val", n_entries=1),
]


def test_write_file_lists(tmp_path):
    output = str(tmp_path)
    write_file_lists(files=FILES, output=output)
    lines = {}
    for dataset in ["train", "val", "test"]:
        with open(os.path.join(output, f"{dataset}.txt")) as f:
            lines[dataset] = f.read().splitlines()
    # The file name only: the directory is left out. A dataset with no files gets an empty list
    assert lines == {"train": ["a.root"], "val": ["b.root"], "test": []}


def test_plan_jobs(tmp_path):
    output = str(tmp_path)
    jobs = plan_jobs(files=FILES, output=output)
    assert sorted(os.listdir(output)) == ["test", "train", "val"]
    assert [
        (job.lstntuple_path, job.lstntuple_entry, job.output_file) for job in jobs
    ] == [
        ("/data/allsim/a.root", 0, os.path.join(output, "train", "a.entry_0.pt")),
        ("/data/allsim/a.root", 1, os.path.join(output, "train", "a.entry_1.pt")),
        ("/data/allsim/b.root", 0, os.path.join(output, "val", "b.entry_0.pt")),
    ]
    # A second production into the same output refuses to start
    with pytest.raises(FileExistsError):
        plan_jobs(files=FILES, output=output)


def test_assign_datasets_exact_sizes():
    datasets = assign_datasets(
        n_files=44,
        train=0.7,
        val=0.15,
        test=0.15,
        seed=42,
    )
    # round(0.15 * 44) = 7
    assert datasets.count("train") == 30
    assert datasets.count("val") == 7
    assert datasets.count("test") == 7


def test_assign_datasets_same_seed_same_datasets():
    first = assign_datasets(n_files=44, train=0.7, val=0.15, test=0.15, seed=1)
    same = assign_datasets(n_files=44, train=0.7, val=0.15, test=0.15, seed=1)
    other = assign_datasets(n_files=44, train=0.7, val=0.15, test=0.15, seed=2)
    assert first == same
    assert first != other


def test_assign_datasets_rejects_fractions_not_summing_to_one():
    with pytest.raises(ValueError):
        assign_datasets(n_files=10, train=0.8, val=0.15, test=0.15, seed=1)
