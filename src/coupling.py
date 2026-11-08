"""Glue between the ANN and the PSO (Task 3).

PSO only knows 'a vector in, a number out'. The ANN only knows 'weights in,
predictions out'. make_fitness joins them: one particle position = one full
set of weights and biases, and its fitness = accuracy on the training data.
"""

from src.metrics import accuracy


def make_fitness(net, X, y):
    """Return fitness(vector) -> training accuracy (higher is better).

    Closure so PSO can call fitness(position) without knowing about net, X or y.
    """

    def fitness(vector):
        net.set_parameters(vector)           # load this particle's weights into the ANN
        return accuracy(y, net.predict(X))   # score it on the TRAINING data only

    return fitness
