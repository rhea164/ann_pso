"""Activation functions used by the ANN (Task 1 of the coursework)."""

import numpy as np


def logistic(x):
    """Logistic (sigmoid) function: 1 / (1 + e^-x). Output is in (0, 1)."""
    # Clip the input so np.exp does not overflow for very large |x|.
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    """ReLU: max(0, x). Output is 0 for negative inputs, x otherwise."""
    return np.maximum(0.0, x)


def tanh(x):
    """Hyperbolic tangent. Output is in (-1, 1)."""
    return np.tanh(x)


# Lookup table so a layer can be configured with a simple string such as "relu".
# Adding a new activation later only needs one more entry here.
ACTIVATIONS = {
    "logistic": logistic,
    "relu": relu,
    "tanh": tanh,
}


def get_activation(name):
    """Return the activation function for a given name, or fail clearly."""
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}"
        )
    return ACTIVATIONS[name]
