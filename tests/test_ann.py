"""Sanity checks for the ANN. Run from the project root: python -m tests.test_ann"""

import numpy as np

from src.ann import NeuralNetwork
from src.activations import logistic, relu, tanh
from src.data import stratified_split


def test_activations():
    # Known values: logistic(0) = 0.5, relu clips negatives, tanh is odd.
    assert logistic(np.array(0.0)) == 0.5
    assert relu(np.array(-3.0)) == 0.0 and relu(np.array(2.0)) == 2.0
    assert tanh(np.array(0.0)) == 0.0
    # Extreme input must not overflow or give NaN.
    assert np.isfinite(logistic(np.array(-1e4)))


def test_parameter_count():
    # [4, 5, 1]: (4*5 + 5) + (5*1 + 1) = 25 + 6 = 31
    net = NeuralNetwork([4, 5, 1], ["tanh", "logistic"])
    assert net.num_parameters() == 31


def test_output_shape_and_range():
    net = NeuralNetwork([4, 6, 3, 1], ["relu", "tanh", "logistic"], seed=1)
    out = net.forward(np.random.default_rng(0).normal(size=(10, 4)))
    assert out.shape == (10, 1)
    assert np.all((out > 0) & (out < 1))  # logistic output layer


def test_parameters_roundtrip():
    # Setting a vector then reading it back must give the same vector.
    net = NeuralNetwork([4, 5, 1], ["tanh", "logistic"])
    vec = np.random.default_rng(2).normal(size=net.num_parameters())
    net.set_parameters(vec)
    assert np.allclose(net.get_parameters(), vec)


def test_known_output():
    # Hand-computed: one layer, 2 inputs, weights [1, 1], bias 0, relu.
    net = NeuralNetwork([2, 1], ["relu"])
    net.set_parameters([1.0, 1.0, 0.0])
    assert net.forward(np.array([[2.0, 3.0]]))[0, 0] == 5.0


def test_bad_inputs_rejected():
    net = NeuralNetwork([4, 5, 1], ["tanh", "logistic"])
    for bad in (lambda: net.set_parameters([1.0, 2.0]),
                lambda: NeuralNetwork([4, 5, 1], ["tanh"]),
                lambda: NeuralNetwork([4, 1], ["sigmoid"])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_stratified_split():
    y = np.array([0] * 80 + [1] * 20)
    X = np.arange(100, dtype=float).reshape(-1, 1)
    _, _, y_tr, y_te = stratified_split(X, y, 0.3, seed=0)
    assert len(y_te) == 30 and len(y_tr) == 70
    assert y_te.mean() == 0.2  # same 20% positive rate as the full data


if __name__ == "__main__":
    # Tiny runner so pytest is not required.
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print("PASS", name)
