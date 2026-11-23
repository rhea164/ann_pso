import numpy as np


def load_banknote(path):
    # each row is 4 features followed by the label (0 genuine, 1 forged)
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :4]
    y = data[:, 4].astype(int)
    return X, y


def stratified_split(X, y, test_ratio=0.3, seed=0):
    # split each class separately so both sets keep the same genuine/forged ratio
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = rng.permutation(train_idx)
    test_idx = rng.permutation(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardise(X_train, X_test):
    # mean and std come from the training set only, so nothing from the test set leaks in
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std
