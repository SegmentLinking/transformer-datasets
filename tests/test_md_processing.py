import awkward as ak
import numpy as np

from transformer_datasets.constants import UNMATCHED
from transformer_datasets.md_processing import (
    hit_sim_pairs,
    inner_is_other,
    md_features,
    md_sim_features,
    order_hits,
)


def test_inner_is_other():
    # barrel MD with the anchor outside, endcap MD with the anchor inside
    anchor = np.array([[30.2, 0.0, 1.0], [50.0, 0.0, 150.0]])
    other = np.array([[30.0, 0.0, 5.0], [40.0, 0.0, 150.4]])
    swap = inner_is_other(anchor=anchor, other=other, layer=np.array([2, 7]))
    assert np.array_equal(swap, [True, False])


def test_hit_sim_pairs():
    # MD 0: anchor hit 10 (sim 5), other hit 11 (no sim): mixed sim + noise
    # MD 1: anchor hit 10 again (shared with MD 0), other hit 12 with two sims
    pairs = hit_sim_pairs(
        anchor_hit=np.array([10, 10]),
        other_hit=np.array([11, 12]),
        anchor_sims=ak.Array([[5], [5]]),
        other_sims=ak.Array([[], [7, 5]]),
    )
    assert np.array_equal(pairs, [[10, 5], [12, 5], [12, 7]])
    assert pairs.dtype == np.int64


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


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"{name}: passed")
