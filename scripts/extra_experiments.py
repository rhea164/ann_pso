import os

import numpy as np
import matplotlib.pyplot as plt

from src.ann import NeuralNetwork
from src.coupling import make_fitness
from src.data import load_banknote, stratified_split, standardise
from src.metrics import accuracy
from src.pso import PSO
from src.stats import paired_permutation_test, bonferroni
from scripts.train_ann_pso import load_data, DATA_PATH

NUM_RUNS = 10
SWARMSIZE = 20
BUDGET = 600
ITERATIONS = BUDGET // SWARMSIZE
LOWER, UPPER = -5, 5
# coefficients adopted from the cognitive/social experiment (beta + gamma + delta = 3, gamma:delta = 2:1)
ALPHA, BETA, GAMMA, DELTA = 0.73, 1.0, 4 / 3, 2 / 3
LAYERS, ACTIVATIONS = [4, 8, 1], ["tanh", "logistic"]   # the ANN is fixed so only the PSO setting changes
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def test_accuracy(net, vector, X_test, y_test):
    net.set_parameters(vector)
    return accuracy(y_test, net.predict(X_test))


def run_pso(data, seed, **pso_args):
    X_train, X_test, y_train, y_test = data
    network = NeuralNetwork(LAYERS, ACTIVATIONS)
    fitness = make_fitness(network, X_train, y_train)
    settings = dict(swarmsize=SWARMSIZE, max_iterations=ITERATIONS, lower=LOWER, upper=UPPER,
                    alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA, boundary_method="clamp")
    settings.update(pso_args)                            # the setting being investigated replaces the baseline value
    np.random.seed(seed)                                 # same seeds for every setting so the comparison is fair
    pso = PSO(fitness, dimensions=network.num_parameters(), **settings)
    best_vector, best_fitness = pso.run()
    return best_fitness, test_accuracy(network, best_vector, X_test, y_test), pso.history


def sweep(title, filename, parameter, values, baseline_value):
    # runs NUM_RUNS times for each value of one PSO argument
    data = load_data()
    train_accs, test_accs, mean_curves = [], [], []

    for value in values:
        runs = [run_pso(data, seed, **{parameter: value}) for seed in range(NUM_RUNS)]
        train_accs.append([r[0] for r in runs])
        test_accs.append([r[1] for r in runs])
        mean_curves.append(np.mean([r[2] for r in runs], axis=0))

    base = values.index(baseline_value)
    print(f"\n{title}  (swarm {SWARMSIZE} x {ITERATIONS}, {NUM_RUNS} runs, baseline {parameter}={baseline_value})")
    print(f"{parameter:>14} | {'train acc':>16} | {'test acc':>16} | {'raw p':>9} | p adj")
    for i, value in enumerate(values):
        if i == base:
            p, p_adj = "-", "-"
        else:
            raw = paired_permutation_test(test_accs[base], test_accs[i])
            p = f"{raw:.3f}"
            p_adj = f"{bonferroni(raw, len(values) - 1):.3f}"   # corrected for the number of comparisons in this table
        print(f"{value:>14} | {np.mean(train_accs[i]):.4f} ± {np.std(train_accs[i]):.4f} | "
              f"{np.mean(test_accs[i]):.4f} ± {np.std(test_accs[i]):.4f} | {p:>9} | {p_adj:>5}")

    plt.figure(figsize=(7, 5))
    plt.errorbar(range(len(values)), [np.mean(t) for t in test_accs],
                 yerr=[np.std(t) for t in test_accs], marker="o", capsize=4)
    plt.xticks(range(len(values)), [str(v) for v in values])
    plt.xlabel(parameter)
    plt.ylabel(f"Test accuracy (mean ± std, {NUM_RUNS} runs)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}_accuracy.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    for value, curve in zip(values, mean_curves):
        plt.plot(range(1, ITERATIONS + 1), curve, label=f"{parameter}={value}")
    plt.xlabel("Iteration")
    plt.ylabel(f"Best training accuracy (mean of {NUM_RUNS} runs)")
    plt.title(f"Convergence: {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}_convergence.png"), dpi=150)
    plt.close()
    print(f"Plots saved in {RESULTS_DIR}/ as {filename}_accuracy.png and {filename}_convergence.png")


def compare_informants():
    # 1 informant = very local, 19 = every other particle in the swarm
    sweep("Number of informants", "pso_informants", "num_informants", [1, 2, 3, 5, 10, 19], 3)


def compare_alpha():
    # alpha = how much velocity is kept (inertia), 0.73 is the usual literature value
    sweep("Inertia (alpha)", "pso_alpha", "alpha", [0.3, 0.5, 0.73, 0.9, 1.0], 0.73)


def compare_splits():
    # the other experiments use one fixed split (seed 0), here every run gets a new split
    fixed = [run_pso(load_data(), seed)[1] for seed in range(NUM_RUNS)]

    X, y = load_banknote(DATA_PATH)
    varying = []
    for seed in range(NUM_RUNS):
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_ratio=0.3, seed=seed)   # new split each run
        X_train, X_test = standardise(X_train, X_test)
        varying.append(run_pso((X_train, X_test, y_train, y_test), seed)[1])

    print(f"\nFixed split vs a different split each run  (baseline network, {NUM_RUNS} runs)")
    print(f"{'fixed split (seed 0)':24} test acc {np.mean(fixed):.4f} ± {np.std(fixed):.4f}")
    print(f"{'new split every run':24} test acc {np.mean(varying):.4f} ± {np.std(varying):.4f} "
          f"(raw p vs fixed = {paired_permutation_test(fixed, varying):.3f}, one comparison so no correction)")

    plt.figure(figsize=(5, 5))
    plt.boxplot([fixed, varying])
    plt.xticks([1, 2], ["fixed split", "new split each run"])
    plt.ylabel("Test accuracy")
    plt.title("Effect of the train/test split")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "split_variability.png"), dpi=150)
    plt.close()
    print(f"Plot saved in {RESULTS_DIR}/split_variability.png")


if __name__ == "__main__":
    compare_informants()
    compare_alpha()
    compare_splits()
