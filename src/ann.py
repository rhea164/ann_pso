import numpy as np

from src.activations import get_activation


class NeuralNetwork:
    # layer_sizes e.g. [4, 8, 1]; activations has one name per layer after the input e.g. ["tanh", "logistic"]

    def __init__(self, layer_sizes, activations, seed=None):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least input and output layers")

        self.layer_sizes = list(layer_sizes)
        self.set_activations(activations)

        # random start weights, biases start at 0 (PSO overwrites them anyway)
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(rng.normal(0.0, 0.5, size=(n_in, n_out)))
            self.biases.append(np.zeros(n_out))

    def set_activations(self, activations):
        if len(activations) != len(self.layer_sizes) - 1:
            raise ValueError(
                f"Expected {len(self.layer_sizes) - 1} activations, got {len(activations)}"
            )
        self.activation_names = list(activations)
        self.activation_funcs = [get_activation(name) for name in activations]

    def forward(self, X):
        # X has one row per banknote, for each layer: multiply by the weights, add the bias,
        # then apply the activation, the result becomes the input of the next layer
        a = X
        for W, b, act in zip(self.weights, self.biases, self.activation_funcs):
            a = act(a @ W + b)
        return a

    def predict(self, X, threshold=0.5):
        # the output node gives a number between 0 and 1 (logistic), 0.5 or more means forged
        output = self.forward(X)
        return (output.ravel() >= threshold).astype(int)

    def num_parameters(self):
        # e.g. [4, 8, 1] has 4*8 + 8 + 8*1 + 1 = 49, which is also the length of a PSO particle
        total = 0
        for W in self.weights:
            total += W.size
        for b in self.biases:
            total += b.size
        return total

    def get_parameters(self):
        # PSO only works on one flat list of numbers, so all weights and biases are
        # flattened into a single vector (one particle = one whole network)
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.append(W.ravel())
            parts.append(b)
        return np.concatenate(parts)

    def set_parameters(self, vector):
        # same order as get_parameters: weights of layer 1, biases of layer 1, then layer 2..
        vector = np.asarray(vector, dtype=float)
        # a wrong length would silently give wrong weights, so stop here
        if vector.size != self.num_parameters():
            raise ValueError(
                f"Expected {self.num_parameters()} parameters, got {vector.size}"
            )

        start = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            self.weights[i] = vector[start:start + W.size].reshape(W.shape)
            start += W.size
            self.biases[i] = vector[start:start + b.size].copy()
            start += b.size
