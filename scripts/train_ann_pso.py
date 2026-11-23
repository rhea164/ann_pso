import sys

import numpy as np

from src.ann import NeuralNetwork
from src.coupling import make_fitness
from src.data import load_banknote, stratified_split, standardise
from src.metrics import accuracy, precision_recall_f1
from src.pso import PSO

DATA_PATH = "data/data_banknote_authentication.txt"

def load_data(path=DATA_PATH):
    # 1. Data: split, then scale with training statistics only.
    X, y = load_banknote(path)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_ratio=0.3, seed=0) #stratified to make sure that each class is adequately represented in each split
    X_train, X_test = standardise(X_train, X_test)
    return X_train, X_test, y_train, y_test

    


def main(path):
    np.random.seed(0)  # PSO uses numpy's global RNG, so this makes the run repeatable

    #1. Load the data
    X_train, X_test, y_train, y_test = load_data(path)
    # 2. ANN whose weights PSO will search over (4 -> 8 -> 1 = 49 numbers).
    net = NeuralNetwork([4, 8, 1], ["tanh", "logistic"])
    fitness = make_fitness(net, X_train, y_train)


    # 3. PSO: each particle has one dimension per ANN parameter.
    pso = PSO(fitness, dimensions=net.num_parameters(), swarmsize=5,
              max_iterations=10, lower=-2.0, upper=2.0)
    best_vector, best_fitness = pso.run()
    
    h=pso.history
    n=len(h)
    checkpoints = [0, n//2, n-1] #start, middle, end

    # 4. Load the best weights found and evaluate on unseen test data.
    # the network currently holds the weights of the last particle evaluated, so this needs to be overwritten with the best one found by PSO.
    net.set_parameters(best_vector)
    print(f"Parameters: {net.num_parameters()} | evaluations: {pso.swarmsize * n}")
    print(f"Best train accuracy from PSO: {best_fitness:.3f}")
    print(f"Fitness after iteration { ' / '.join(str(i+1) for i in checkpoints)}: "
          f"{' / '.join(f'{h[i]:.3f}' for i in checkpoints)}")
    for name, Xs, ys in (("train", X_train, y_train), ("test", X_test, y_test)):
        pred = net.predict(Xs)
        p, r, f1 = precision_recall_f1(ys, pred)
        print(f"{name}: acc={accuracy(ys, pred):.3f} precision={p:.3f} recall={r:.3f} f1={f1:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_PATH)
