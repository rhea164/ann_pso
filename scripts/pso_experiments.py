import os

import numpy as np
import matplotlib.pyplot as plt

from src.pso import PSO

NUM_RUNS = 10
SWARMSIZE = 30
MAX_ITERATIONS = 100
LOWER,UPPER = -5,5
RESULTS_DIR = "results"  # plots are saved here (kept out of the project root)
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.ann import NeuralNetwork
from src.coupling import make_fitness
from src.data import load_banknote, stratified_split, standardise
from src.metrics import accuracy, precision_recall_f1
from src.pso import PSO
from scripts.train_ann_pso import load_data

def test_accuracy(net, vector, X_test, y_test):
    """Load a weight vector into the network and measure accuracy on unseen data."""
    net.set_parameters(vector)
    return accuracy(y_test, net.predict(X_test))
    

def compare_boundary_methods():
    """Going further: Compare no boundary handling, clamping and random reset"""
    network = NeuralNetwork([4, 8, 1], ["tanh", "logistic"])
    X_train, X_test, y_train, y_test = load_data()
    fitness = make_fitness(network, X_train, y_train)
    results={}
    for method in [None, "clamp", "random_reset"]:
        test_accuracies, curves, escapes=[],[],[]
        for run in range(NUM_RUNS):
            np.random.seed(run)  # repeatable runs
            pso = PSO(fitness, dimensions=network.num_parameters(),swarmsize =SWARMSIZE, max_iterations=MAX_ITERATIONS, lower=LOWER, upper=UPPER, boundary_method=method)
            best_vector, best_fitness = pso.run()
            test_accuracies.append(test_accuracy(network, best_vector, X_test, y_test))
            curves.append(pso.history)
            escapes.append(pso.escapes) 
            
        results[method] = curves
        print(f"{str(method):13} test acc {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f} | "
              f"escapes {np.mean(escapes):.0f}")
    # Average convergence curve for each method
    for method, curves in results.items():
        plt.plot(np.mean(curves, axis=0), label=str(method))
    plt.xlabel("Iteration")
    plt.ylabel(f"Best training accuracy (mean of {NUM_RUNS} runs)")
    plt.title("Boundary handling comparison")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "boundary_comparison.png"), dpi=150)
    plt.close()
    print("Plot saved to boundary_comparison.png")

def compare_swarm_sizes(budget=600):
    """Task 5 Q2: for a fixed budget of evaluations, is a big swarm with few
    iterations better than a small swarm with many iterations?"""
    network = NeuralNetwork([4, 8, 1], ["tanh", "logistic"]) #freezing the ann to isolate the effect of swarm size 
    X_train, X_test, y_train, y_test = load_data()
    fitness = make_fitness(network, X_train, y_train)
    swarm_sizes = [5, 10, 20, 30, 50, 100] # different swarm sizes for testing
    means, stds = [], []

    for size in swarm_sizes:
        iterations = budget // size        # budget = swarm size x iterations 
        test_accuracies, train_accuracies, curves = [], [], []

        for run in range(NUM_RUNS):
            np.random.seed(run)            # same seeds for every swarm size in the corresponding runs so that any difference
                                           # in performance is due to swarm size and not random initialisation
            pso = PSO(fitness, dimensions=network.num_parameters(), swarmsize=size,
                      max_iterations=iterations, lower=LOWER, upper=UPPER,
                      boundary_method="clamp")
            best_vector, best_fitness = pso.run()

            train_accuracies.append(best_fitness)
            test_accuracies.append(test_accuracy(network, best_vector, X_test, y_test))
            # curves stores the best training accuracy at each iteration for each run. each row is a run, each column is an iteration
            curves.append(pso.history)

        # Store the mean and std of test accuracies for this swarm size
        means.append(np.mean(test_accuracies))
        stds.append(np.std(test_accuracies))
        print(f"swarm {size:3} x {iterations:3} iterations | "
              f"train acc {np.mean(train_accuracies):.4f} | "
              f"test acc {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies   ):.4f}")

        # Convergence plotted against evaluations
        #EVALUATIONS and not iterations because one iteration costs different amt of work for different swarm sizes and would not be comparable.
        # The total evaluations is same for all swarm sizes, but the number of iterations is different. So we plot against evaluations to compare convergence.
        evaluations = [(i + 1) * size for i in range(iterations)]
        # we average the curves across runs to get the mean best training accuracy at each evaluation for this swarm size
        plt.plot(evaluations, np.mean(curves, axis=0), label=f"swarm {size}")

    plt.xlabel("Fitness evaluations")
    plt.ylabel(f"Best training accuracy (mean of {NUM_RUNS} runs)")
    plt.title(f"Convergence for each swarm size (budget {budget})")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"swarm_size_convergence_{budget}.png"), dpi=150)
    plt.close()

    # Final test accuracy against swarm size, with error bars of one std
    plt.errorbar(swarm_sizes, means, yerr=stds, marker="o", capsize=4)
    plt.xscale("log")
    plt.xticks(swarm_sizes, [str(s) for s in swarm_sizes])
    plt.xlabel("Swarm size (iterations = budget / swarm size)")
    plt.ylabel(f"Test accuracy (mean ± std, {NUM_RUNS} runs)")
    plt.title(f"Swarm size vs iterations (budget {budget})")
    plt.savefig(os.path.join(RESULTS_DIR, f"swarm_size_accuracy_{budget}.png"), dpi=150)
    plt.close()
    print(f"Plots saved to swarm_size_convergence_{budget}.png and swarm_size_accuracy_{budget}.png")

def compare_coefficients(budget=600,swarmsize=20):
    """Task 5 Q3: compare different coefficients for the velocity update equation.
    Balance between cognitive (beta) and social (gamma,delta) components.
    Total beta+gamma+delta is constant at 3, so only the balance between them changes.
    gamme to delta is fixed at ratio 2:1.
    """
    network = NeuralNetwork([4, 8, 1], ["tanh", "logistic"])
    X_train, X_test, y_train, y_test = load_data()
    fitness = make_fitness(network, X_train, y_train)
    iterations = budget // swarmsize
    TOTAL = 3.0
    beta_values = [0.0,0.5,1.0,1.5,2.0,2.5,3.0] # 0:only social, 3: only cognitive 
    means, stds = [], []
    for beta in beta_values:
        social = TOTAL - beta
        gamma = 2/3 * social
        delta = 1/3 * social
        test_accuracies, train_accuracies, curves = [], [], []
        for run in range(NUM_RUNS):
            np.random.seed(run)
            pso = PSO(fitness, dimensions=network.num_parameters(), swarmsize=swarmsize,
                      max_iterations=iterations, lower=LOWER, upper=UPPER,
                      alpha=0.73, beta=beta, gamma=gamma, delta=delta,
                      boundary_method="clamp")
            best_vector, best_fitness = pso.run()
            train_accuracies.append(best_fitness)
            test_accuracies.append(test_accuracy(network, best_vector, X_test, y_test))
            curves.append(pso.history)
        means.append(np.mean(test_accuracies))
        stds.append(np.std(test_accuracies))
        print(f"beta {beta:.1f} | gamma {gamma:.2f} | delta {delta:.2f} | "
              f"train acc {np.mean(train_accuracies):.4f} | "
              f"test acc {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
        plt.plot(np.mean(curves, axis=0), label=f"beta {beta:.1f}, social {social:.1f}")
        
    plt.xlabel("Iteration")
    plt.ylabel(f"Best training accuracy (mean of {NUM_RUNS} runs)")
    plt.title(f"Convergence for different coefficients (cognitive / social balance)")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"coefficients_convergence_{budget}.png"), dpi=150)
    plt.close()
    
    plt.errorbar(beta_values, means, yerr=stds, marker="o", capsize=4)
    plt.xticks(beta_values, [f"{b:.1f}\n{TOTAL - b:.1f}" for b in beta_values])
    plt.xlabel("Cognitive beta (top) / Social gamma+delta (bottom)")
    plt.ylabel(f"Test accuracy (mean ± std, {NUM_RUNS} runs)")
    plt.title(f"Effect of cognitive/social balance on test accuracy")
    plt.savefig(os.path.join(RESULTS_DIR, f"coefficients_accuracy_{budget}.png"), dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # compare_boundary_methods()
    # compare_swarm_sizes(300)
    compare_coefficients()
    
    