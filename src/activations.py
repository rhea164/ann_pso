import numpy as np


def logistic(x):
    x = np.clip(x, -500, 500)  # stops np.exp overflowing
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0.0, x)


def leaky_relu(x, slope=0.01):
    return np.where(x > 0, x, slope * x)


def tanh(x):
    return np.tanh(x)


# activations by name so a layer can be set with a string like "relu"
ACTIVATIONS = {
    "logistic": logistic,
    "relu": relu,
    "tanh": tanh,
    "leaky_relu": leaky_relu,
}


def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}"
        )
    return ACTIVATIONS[name]
