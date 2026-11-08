"""Feedforward multi-layer ANN (Task 1).

There is deliberately no training code here (no backpropagation). The weights
and biases are set from outside, by PSO, through set_parameters().
"""

import numpy as np

from src.activations import get_activation


class NeuralNetwork:
    """Fully connected feedforward network.

    layer_sizes: nodes per layer including input and output, e.g. [4, 8, 1].
    activations: one activation name per non-input layer, e.g. ["tanh", "logistic"].
    """

    def __init__(self, layer_sizes, activations, seed=None):
        # An ANN needs at least an input and an output layer.
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least input and output layers")

        self.layer_sizes = list(layer_sizes)
        self.set_activations(activations)

        # Start with small random weights and zero biases. PSO will overwrite
        # these, but this means the network is usable straight away (and lets us
        # test the evaluation code on random weights, as the spec suggests).
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(rng.normal(0.0, 0.5, size=(n_in, n_out)))
            self.biases.append(np.zeros(n_out))

    def set_activations(self, activations):
        """Set the activation of each layer (kept separate so PSO can change it later)."""
        # One activation per layer after the input layer (the input layer does no computation).
        if len(activations) != len(self.layer_sizes) - 1:
            raise ValueError(
                f"Expected {len(self.layer_sizes) - 1} activations, got {len(activations)}"
            )
        self.activation_names = list(activations)
        self.activation_funcs = [get_activation(name) for name in activations]

    def forward(self, X):
        """Run a batch of inputs X (shape: samples x features) through the network."""
        a = X
        for W, b, act in zip(self.weights, self.biases, self.activation_funcs):
            # Each layer: weighted sum of previous outputs plus bias, then activation.
            a = act(a @ W + b)
        return a

    def predict(self, X, threshold=0.5):
        """Convert the single output node into a 0/1 class label."""
        output = self.forward(X)
        # Flatten (samples x 1) to (samples,) so it compares directly with the labels.
        return (output.ravel() >= threshold).astype(int)

    # ---- Flat parameter vector: this is what PSO will optimise ----

    def num_parameters(self):
        """Total number of weights and biases (= length of a PSO particle)."""
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def get_parameters(self):
        """Flatten all weights and biases into one 1D vector."""
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.append(W.ravel())
            parts.append(b)
        return np.concatenate(parts)

    def set_parameters(self, vector):
        """Load a flat vector back into the layers (same order as get_parameters)."""
        vector = np.asarray(vector, dtype=float)
        # Wrong length would silently mis-assign weights, so fail loudly instead.
        if vector.size != self.num_parameters():
            raise ValueError(
                f"Expected {self.num_parameters()} parameters, got {vector.size}"
            )

        start = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Take the next W.size values and reshape them into the weight matrix.
            self.weights[i] = vector[start:start + W.size].reshape(W.shape)
            start += W.size
            # Then the next b.size values are this layer's biases.
            self.biases[i] = vector[start:start + b.size].copy()
            start += b.size
