from src.metrics import accuracy


def make_fitness(net, X, y):
    # This is how PSO and the ANN are coupled. PSO only needs a function that takes a
    # particle's position (a vector) and returns a score, so we build one that:
    # loads the vector into the ANN as weights and biases, runs the training data
    # through it, and returns the accuracy. PSO tries to maximise that accuracy
    def fitness(vector):
        net.set_parameters(vector)           # put this particle's weights into the ANN
        return accuracy(y, net.predict(X))   # fitness = accuracy on the training data

    return fitness


# Extension: PSO also chooses the hidden-layer activations
# The particle gets one extra number per hidden layer (after the weights and biases),
# and that number picks the activation, the output layer stays logistic
ACTIVATION_NAMES = ["logistic", "relu", "tanh"]  # search range is split into equal parts, one per activation


def decode_activations(genes, lower, upper):
    names = []
    for g in genes:
        fraction = (g - lower) / (upper - lower)                # position in the search range, 0 to 1
        index = int(fraction * len(ACTIVATION_NAMES))
        index = min(max(index, 0), len(ACTIVATION_NAMES) - 1)   # keeps it valid if g is exactly on a bound
        names.append(ACTIVATION_NAMES[index])
    return names


def load_vector_with_activations(net, vector, lower, upper):
    n_weights = net.num_parameters()
    net.set_parameters(vector[:n_weights])
    hidden = decode_activations(vector[n_weights:], lower, upper)
    net.set_activations(hidden + ["logistic"])


def make_fitness_with_activations(net, X, y, lower, upper):
    # PSO dimensions needed = net.num_parameters() + number of hidden layers
    def fitness(vector):
        load_vector_with_activations(net, vector, lower, upper)
        return accuracy(y, net.predict(X))

    return fitness
