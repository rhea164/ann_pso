"""Evaluate a random-weight ANN on the banknote data.

Run from the project root:  python -m scripts.check_ann [path_to_data]
Accuracy of random weights varies wildly (anywhere from ~5% to ~95%); the baseline to beat is 55.5%.
"""

import sys

from src.ann import NeuralNetwork
from src.data import load_banknote, stratified_split, standardise
from src.metrics import accuracy, precision_recall_f1

DATA_PATH = "data/data_banknote_authentication.txt"


def main(path):
    # 1. Load and split the data, then scale using training statistics only.
    X, y = load_banknote(path)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_ratio=0.3, seed=0)
    X_train, X_test = standardise(X_train, X_test)
    print(f"Samples: {len(y)} | train: {len(y_train)} | test: {len(y_test)}")
    print(f"Forged rate  train: {y_train.mean():.3f}  test: {y_test.mean():.3f}")

    # 2. Build an ANN: 4 inputs -> 8 hidden (tanh) -> 1 output (logistic).
    net = NeuralNetwork([4, 8, 1], ["tanh", "logistic"], seed=42)
    print(f"Network parameters: {net.num_parameters()}")

    # 3. Evaluate the untrained network on both splits.
    for name, Xs, ys in (("train", X_train, y_train), ("test", X_test, y_test)):
        pred = net.predict(Xs)
        p, r, f1 = precision_recall_f1(ys, pred)
        print(f"{name}: acc={accuracy(ys, pred):.3f} precision={p:.3f} recall={r:.3f} f1={f1:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_PATH)
