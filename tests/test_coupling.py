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


def test_decode_activations_bins():
    # lower end -> first activation, upper end -> last, middle -> middle one
    from src.coupling import decode_activations, ACTIVATION_NAMES
    names = decode_activations([-5.0, 0.0, 5.0, 99.0], -5.0, 5.0)
    assert names == [ACTIVATION_NAMES[0], ACTIVATION_NAMES[1], ACTIVATION_NAMES[-1], ACTIVATION_NAMES[-1]]


def test_vector_with_activations_sets_network():
    # the extra genes must change the network's hidden activations, output stays logistic
    from src.coupling import load_vector_with_activations
    net, X, y = make_toy()
    vec = np.concatenate([net.get_parameters(), [5.0]])  # one hidden layer, gene at the top of the range
    load_vector_with_activations(net, vec, -5.0, 5.0)
    assert net.activation_names == ["tanh", "logistic"]
    vec[-1] = -5.0
    load_vector_with_activations(net, vec, -5.0, 5.0)
    assert net.activation_names == ["logistic", "logistic"]


def test_fitness_with_activations_in_range():
    from src.coupling import make_fitness_with_activations
    net, X, y = make_toy()
    vec = np.concatenate([net.get_parameters(), [0.0]])
    assert 0.0 <= make_fitness_with_activations(net, X, y, -5.0, 5.0)(vec) <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
