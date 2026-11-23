import numpy as np


def permutation_test(a, b, n_permutations=10000, seed=0):
    # two-sided test on the difference of the means: shuffle the runs of both
    # settings together and see how often a difference this big happens by chance
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        diff = abs(pooled[:len(a)].mean() - pooled[len(a):].mean())
        if diff >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)  # +1 so p is never exactly 0


def paired_permutation_test(a, b, n_permutations=10000, seed=0):
    # a[i] and b[i] come from the same seed (run i), so they are a pair
    # two-sided test on the mean difference: if the settings made no difference, the
    # sign of each pair's difference is arbitrary, so we flip the signs at random
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("a and b must be 1-D and have the same length (one value per seed)")
    if len(a) == 0:
        raise ValueError("need at least one pair")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("a and b must not contain nan or inf")

    differences = a - b
    observed = abs(differences.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(differences)))
    permuted = np.abs((signs * differences).mean(axis=1))
    count = np.sum(permuted >= observed - 1e-12)   # tiny tolerance so rounding does not break ties
    return (count + 1) / (n_permutations + 1)      # the observed result counts as one permutation, so p is never 0


def bonferroni(p, n_comparisons):
    # several settings compared with one baseline, so multiply p by the number of comparisons
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be at least 1")
    return min(1.0, p * n_comparisons)
