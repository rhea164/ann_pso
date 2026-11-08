"""Loading and preparing the banknote authentication dataset (Task 4)."""

import numpy as np


def load_banknote(path):
    """Read the CSV. First 4 columns are features, last column is the 0/1 label."""
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :4]
    y = data[:, 4].astype(int)
    return X, y


def stratified_split(X, y, test_ratio=0.3, seed=0):
    """Split into train/test so both splits keep the original class proportions."""
    # A fixed seed makes the split reproducible between runs.
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    # Split each class separately; this is what makes the split "stratified".
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    # Shuffle again so the classes are not stored in blocks.
    train_idx = rng.permutation(train_idx)
    test_idx = rng.permutation(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardise(X_train, X_test):
    """Scale features to mean 0, std 1 using the TRAINING statistics only."""
    # Using test data here would leak information about it into training.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std
