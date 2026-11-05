from constants import UNMATCHED
from t3_processing import resolve_t3


def test_resolve_t3_all_same():
    hits = [[4], [4], [4], [4], [4], [4]]
    assert resolve_t3(hits) == [4, 4, 4, 4, 4, 4]


def test_resolve_t3_docstring_example():
    hits = [[7], [7], [3, 7], [3], [], [7, 3]]
    assert resolve_t3(hits) == [7, 7, 7, 3, UNMATCHED, 7]


def test_resolve_t3_single_sim_indices_unchanged():
    hits = [[5], [6], [5], [], [6], [8]]
    assert resolve_t3(hits) == [5, 6, 5, UNMATCHED, 6, 8]


def test_resolve_t3_tie_broken_by_first_appearance_for_every_hit():
    # 1 and 2 each appear in two hits; 1 appears first, so both hits take 1
    hits = [[1, 2], [2, 1]]
    assert resolve_t3(hits) == [1, 1]


def test_resolve_t3_duplicate_sim_index_in_one_hit_counted_once():
    # 4 appears twice in the first hit but in only one hit overall, so 9 (two hits) wins
    hits = [[4, 4, 9], [9]]
    assert resolve_t3(hits) == [9, 9]


def test_resolve_t3_all_unmatched():
    assert resolve_t3([[]] * 6) == [UNMATCHED] * 6


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"{name}: passed")
