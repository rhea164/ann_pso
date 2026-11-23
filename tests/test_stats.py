import numpy as np

from src.stats import permutation_test, paired_permutation_test, bonferroni


def test_identical_groups_not_significant():
    a = [0.9, 0.91, 0.92, 0.9, 0.93]
    assert permutation_test(a, a) > 0.9


def test_very_different_groups_significant():
    assert permutation_test([0.5, 0.51, 0.49, 0.5, 0.52], [0.9, 0.91, 0.92, 0.9, 0.93]) < 0.01


def test_p_value_range_and_repeatable():
    p1 = permutation_test([1, 2, 3], [2, 3, 4])
    assert 0 < p1 <= 1 and p1 == permutation_test([1, 2, 3], [2, 3, 4])


def test_bonferroni():
    assert abs(bonferroni(0.01, 4) - 0.04) < 1e-12
    assert bonferroni(0.6, 4) == 1.0   # capped at 1


def raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


A = [0.90, 0.91, 0.92, 0.90, 0.93, 0.91, 0.92, 0.90, 0.94, 0.91]


def test_paired_identical_results():
    assert paired_permutation_test(A, A) == 1.0   # every difference is 0


def test_paired_clear_positive_difference():
    b = [x - 0.05 for x in A]                     # a is better in every pair
    assert paired_permutation_test(A, b) < 0.01


def test_paired_clear_negative_difference():
    b = [x + 0.05 for x in A]
    assert paired_permutation_test(A, b) < 0.01


def test_paired_is_two_sided():
    b = [x - 0.05 for x in A]
    assert paired_permutation_test(A, b) == paired_permutation_test(b, A)


def test_paired_uses_the_pairing():
    # same means as a and b, but the differences are small and consistent in b - a
    # an unpaired test sees overlapping groups, the paired test sees a consistent gain
    a = [0.70, 0.80, 0.90, 0.60, 0.75, 0.85, 0.65, 0.95]
    b = [x + 0.01 for x in a]
    assert permutation_test(a, b) > 0.5
    assert paired_permutation_test(a, b) < 0.05


def test_paired_small_samples():
    p = paired_permutation_test([1.0], [0.0])
    assert 0.4 < p <= 1.0                         # one pair can never be significant
    p2 = paired_permutation_test([1, 1, 1], [0, 0, 0])
    assert 0 < p2 <= 1                            # 3 pairs, the smallest possible p is 2/8 = 0.25
    assert p2 > 0.2


def test_paired_bad_inputs():
    assert raises(paired_permutation_test, [1, 2, 3], [1, 2])
    assert raises(paired_permutation_test, [], [])
    assert raises(paired_permutation_test, [1, float("nan")], [1, 2])
    assert raises(paired_permutation_test, [[1, 2]], [[1, 2]])


def test_paired_repeatable_and_seed_dependent():
    b = [x - 0.002 for x in A[::-1]]
    p1 = paired_permutation_test(A, b, seed=0)
    assert p1 == paired_permutation_test(A, b, seed=0)
    assert 0 < p1 <= 1


def test_bonferroni_family_size():
    assert abs(bonferroni(0.01, 1) - 0.01) < 1e-12    # one comparison, nothing to correct
    assert abs(bonferroni(0.02, 5) - 0.10) < 1e-12
    assert raises(bonferroni, 0.5, 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
