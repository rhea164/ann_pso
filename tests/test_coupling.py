"""Sanity tests for the ANN <-> PSO coupling. Run: python -m tests.test_coupling"""

import numpy as np

from src.ann import NeuralNetwork
from src.coupling import make_fitness


def make_toy():
    # 4 tiny samples, 2 features, labels 0/1
    X = np.array([[-1.0, -1.0], [-0.5, -1.0], [1.0, 1.0], [0.5, 1.0]])
    y = np.array([0, 0, 1, 1])
    net = NeuralNetwork([2, 3, 1], ["tanh", "logistic"], seed=0)
    return net, X, y


def test_fitness_in_range():
    # accuracy is a fraction, so it must lie in [0, 1]
    net, X, y = make_toy()
    f = make_fitness(net, X, y)(net.get_parameters())
    assert 0.0 <= f <= 1.0


def test_fitness_matches_manual_accuracy():
    # fitness must equal accuracy computed by hand with the same weights
    net, X, y = make_toy()
    vec = np.random.RandomState(1).normal(size=net.num_parameters())
    f = make_fitness(net, X, y)(vec)
    net.set_parameters(vec)
    assert f == np.mean(net.predict(X) == y)


def test_fitness_depends_on_vector():
    # different weight vectors must be able to give different scores,
    # otherwise PSO would have nothing to optimise
    net, X, y = make_toy()
    fit = make_fitness(net, X, y)
    rng = np.random.RandomState(2)
    scores = {fit(rng.normal(size=net.num_parameters())) for _ in range(50)}
    assert len(scores) > 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
