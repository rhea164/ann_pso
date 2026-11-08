# F20BC/F21BC Coursework – ANN trained with PSO

Implemented: ANN, PSO (informants, Algorithm 39) and the coupling that trains the ANN with PSO. Experiments to follow.

## Setup
```
pip install -r requirements.txt
```
Download the banknote dataset (see below) and place
`data_banknote_authentication.txt` in `data/`.

## Run
From the project root:
```
python -m tests.test_ann        # ANN sanity checks
python -m tests.test_coupling   # fitness (ANN <-> PSO) checks
python -m scripts.check_ann     # random-weight ANN evaluated on the data
python -m scripts.train_ann_pso # train the ANN with PSO once, report train/test
```

## Layout
| File | Purpose |
|------|---------|
| `src/activations.py` | logistic, ReLU, tanh |
| `src/ann.py` | `NeuralNetwork` class (forward pass, flat parameter get/set) |
| `src/data.py` | load CSV, stratified 70/30 split, standardisation |
| `src/metrics.py` | accuracy, precision, recall, F1 |
| `src/particle.py` | one PSO particle (velocity update, move) |
| `src/pso.py` | PSO swarm and main loop (Algorithm 39) |
| `src/coupling.py` | fitness(vector) = ANN training accuracy |
| `tests/test_ann.py` | ANN sanity tests |
| `tests/test_coupling.py` | coupling (fitness) tests |
| `scripts/check_ann.py` | end-to-end check with random weights |
| `scripts/train_ann_pso.py` | one PSO-trained run |
| `notebooks/01_data_exploration.ipynb` | charts and checks of the banknote data (needs jupyter, optional) |
