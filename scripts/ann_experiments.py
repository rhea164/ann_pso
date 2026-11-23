import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from src.ann import NeuralNetwork
from src.coupling import make_fitness, make_fitness_with_activations, load_vector_with_activations
from src.metrics import accuracy
from src.pso import PSO
from src.stats import paired_permutation_test, bonferroni
from scripts.train_ann_pso import load_data

NUM_RUNS = 10          # at least 10 runs for each setting (spec)
SWARMSIZE = 20         # same as the coefficient experiment so the results can be compared
BUDGET = 600           # swarm size x iterations
ITERATIONS = BUDGET // SWARMSIZE
LOWER, UPPER = -5, 5
# coefficients adopted from the cognitive/social experiment (beta + gamma + delta = 3, gamma:delta = 2:1)
ALPHA, BETA, GAMMA, DELTA = 0.73, 1.0, 4 / 3, 2 / 3
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def test_accuracy(net, vector, X_test, y_test):
    net.set_parameters(vector)
    return accuracy(y_test, net.predict(X_test))


def run_architecture(layers, activations, data):
    X_train, X_test, y_train, y_test = data
    # only the network changes between designs, the PSO settings below are the same every time
    network = NeuralNetwork(layers, activations)         # PSO overwrites the weights, so the start values do not matter
    fitness = make_fitness(network, X_train, y_train)
    train_accuracies, test_accuracies, curves = [], [], []

    for run in range(NUM_RUNS):
        np.random.seed(run)                              # same seeds for every design so the comparison is fair
        pso = PSO(fitness, dimensions=network.num_parameters(), swarmsize=SWARMSIZE,
                  max_iterations=ITERATIONS, lower=LOWER, upper=UPPER,
                  alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
                  boundary_method="clamp")
        best_vector, best_fitness = pso.run()
        train_accuracies.append(best_fitness)
        test_accuracies.append(test_accuracy(network, best_vector, X_test, y_test))
        curves.append(pso.history)                       # best training accuracy after each iteration

    return network.num_parameters(), train_accuracies, test_accuracies, curves


def run_designs(title, filename, designs, baseline=0, connect=True):
    # every design is compared with designs[baseline]
    # connect=False draws points only, for designs that have no order (e.g. different activation functions)
    data = load_data()
    labels, params, train_accs, test_accs, mean_curves = [], [], [], [], []

    for label, layers, activations in designs:
        n_params, train, test, curves = run_architecture(layers, activations, data)
        labels.append(label)
        params.append(n_params)
        train_accs.append(train)
        test_accs.append(test)
        mean_curves.append(np.mean(curves, axis=0))      # average over the runs for the convergence plot

    print(f"\n{title}  (swarm {SWARMSIZE} x {ITERATIONS} iterations = budget {BUDGET}, {NUM_RUNS} runs)")
    print(f"{'setting':24} {'params':>6} | {'train acc':>16} | {'test acc':>16} | {'raw p':>9} | p adj")
    # paired permutation test against the baseline (run i of every design uses seed i), then Bonferroni because several designs are compared
    for i, label in enumerate(labels):
        if i == baseline:
            p, p_adj = "-", "-"
        else:
            raw = paired_permutation_test(test_accs[baseline], test_accs[i])
            p = f"{raw:.3f}"
            p_adj = f"{bonferroni(raw, len(labels) - 1):.3f}"   # corrected for the number of comparisons in this table
        print(f"{label:24} {params[i]:6d} | {np.mean(train_accs[i]):.4f} ± {np.std(train_accs[i]):.4f} | "
              f"{np.mean(test_accs[i]):.4f} ± {np.std(test_accs[i]):.4f} | {p:>9} | {p_adj:>5}")

    # Final test accuracy for each design, error bar = one standard deviation
    plt.figure(figsize=(8, 5))
    plt.errorbar(range(len(labels)), [np.mean(t) for t in test_accs],
                 yerr=[np.std(t) for t in test_accs], marker="o", capsize=4,
                 linestyle="-" if connect else "")
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel(f"Test accuracy (mean ± std, {NUM_RUNS} runs)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}_accuracy.png"), dpi=150)
    plt.close()

    # How fast each design improves during the run
    plt.figure(figsize=(8, 5))
    for label, curve in zip(labels, mean_curves):
        plt.plot(range(1, ITERATIONS + 1), curve, label=label)
    plt.xlabel("Iteration")
    plt.ylabel(f"Best training accuracy (mean of {NUM_RUNS} runs)")
    plt.title(f"Convergence: {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}_convergence.png"), dpi=150)
    plt.close()
    print(f"Plots saved in {RESULTS_DIR}/ as {filename}_accuracy.png and {filename}_convergence.png")


def compare_hidden_neurons():
    # Q1a: how many neurons in one hidden layer
    run_designs("Hidden layer size", "ann_hidden_neurons", [
        ("[4,2,1]", [4, 2, 1], ["tanh", "logistic"]),
        ("[4,4,1]", [4, 4, 1], ["tanh", "logistic"]),
        ("[4,8,1] (baseline)", [4, 8, 1], ["tanh", "logistic"]),
        ("[4,16,1]", [4, 16, 1], ["tanh", "logistic"]),
    ], baseline=2)


def compare_depth():
    # Q1b: number of hidden layers
    run_designs("Number of hidden layers", "ann_depth", [
        ("1 hidden [4,8,1]", [4, 8, 1], ["tanh", "logistic"]),
        ("2 hidden [4,8,4,1]", [4, 8, 4, 1], ["tanh", "tanh", "logistic"]),
        ("3 hidden [4,8,8,4,1]", [4, 8, 8, 4, 1], ["tanh", "tanh", "tanh", "logistic"]),
    ])


def compare_activations():
    # Q1c: hidden activation function (output layer stays logistic)
    layers = [4, 8, 1]
    run_designs("Hidden activation function", "ann_activation", [
        ("tanh (baseline)", layers, ["tanh", "logistic"]),
        ("relu", layers, ["relu", "logistic"]),
        ("logistic", layers, ["logistic", "logistic"]),
        ("leaky_relu", layers, ["leaky_relu", "logistic"]),
    ], connect=False)


def compare_evolved_activations():
    # going further: PSO picks the hidden activations itself, compared with always using tanh
    X_train, X_test, y_train, y_test = load_data()
    print(f"\nFixed tanh vs PSO-chosen activations  (swarm {SWARMSIZE} x {ITERATIONS}, {NUM_RUNS} runs)")
    print(f"{'network':14} {'activations':12} | {'train acc':>16} | {'test acc':>16} | chosen by PSO (raw p, p adj for 2 comparisons)")
    labels, means, stds = [], [], []

    for layers in ([4, 8, 1], [4, 8, 4, 1]):
        n_hidden = len(layers) - 2
        start_activations = ["tanh"] * n_hidden + ["logistic"]

        # (a) baseline: tanh in every hidden layer, PSO only searches the weights
        _, fixed_train, fixed_test, _ = run_architecture(layers, start_activations,
                                                         (X_train, X_test, y_train, y_test))
        print(f"{str(layers):14} {'fixed tanh':12} | {np.mean(fixed_train):.4f} ± {np.std(fixed_train):.4f} | "
              f"{np.mean(fixed_test):.4f} ± {np.std(fixed_test):.4f} | -")
        labels.append(f"{layers} fixed")
        means.append(np.mean(fixed_test))
        stds.append(np.std(fixed_test))

        # (b) extension: each particle has one extra number per hidden layer that picks its activation,
        # so PSO searches the weights and the activations at the same time
        network = NeuralNetwork(layers, start_activations)
        fitness = make_fitness_with_activations(network, X_train, y_train, LOWER, UPPER)
        dimensions = network.num_parameters() + n_hidden
        train_accuracies, test_accuracies, chosen = [], [], []
        for run in range(NUM_RUNS):
            np.random.seed(run)                          # same seeds as the fixed version
            pso = PSO(fitness, dimensions=dimensions, swarmsize=SWARMSIZE, max_iterations=ITERATIONS,
                      lower=LOWER, upper=UPPER, alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
                      boundary_method="clamp")
            best_vector, best_fitness = pso.run()
            load_vector_with_activations(network, best_vector, LOWER, UPPER)   # weights and activations of the best particle
            train_accuracies.append(best_fitness)
            test_accuracies.append(accuracy(y_test, network.predict(X_test)))
            chosen.append(tuple(network.activation_names[:-1]))                # hidden layers only
        p = paired_permutation_test(fixed_test, test_accuracies)   # same seeds in both versions
        print(f"{str(layers):14} {'PSO-chosen':12} | {np.mean(train_accuracies):.4f} ± {np.std(train_accuracies):.4f} | "
              f"{np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f} | "
              f"{Counter(chosen).most_common(3)} (raw p={p:.3f}, adj p={bonferroni(p, 2):.3f})")
        labels.append(f"{layers} PSO-chosen")
        means.append(np.mean(test_accuracies))
        stds.append(np.std(test_accuracies))

    plt.figure(figsize=(8, 5))
    plt.errorbar(range(len(labels)), means, yerr=stds, fmt="o", capsize=4)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel(f"Test accuracy (mean ± std, {NUM_RUNS} runs)")
    plt.title("Fixed tanh vs PSO-chosen hidden activations")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ann_evolved_activations_accuracy.png"), dpi=150)
    plt.close()
    print(f"Plot saved in {RESULTS_DIR}/ann_evolved_activations_accuracy.png")


if __name__ == "__main__":
    compare_hidden_neurons()
    compare_depth()
    compare_activations()
    compare_evolved_activations()
