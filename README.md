# Error Mitigation

This project corresponds to Project 4 on error mitigation.

# Structure

## figs_...

It contains all pdf/png files used in the presentation

## results

It contains the results of the quantum circuit simulations:
- `3qb...` : 3-qubit circuits
- `...4/5_qubits`: 4- and 5-qubit circuits (with exponential ZNE)
- `fakeperth`: use the FakePerth noise model for the simulation, and 1% depolarizing noise otherwise
- `circuit`: circuit folding
- `layer`: layer folding
- `exp` / `lin` / `quad` / `rich` : respectively exponential, linear, quadratic and Richardson extrapolation method for the ZNE


Each subdirectory is decomposed as follows:
- `circuits` folder : contains all circuits used for the folding procedure at various noise levels, both in `.pdf` and `.qc` format. `.qc` means that this is a `QuantumCircuit` object saved with pickle
- `rep_i` folders : each correspond to 1 replication of the computation (1 "experiment"). Subsubdirectories:
    - `count`: dictionary saved as pickle files. Contains the count of each binary outcome when running `execute(qc).result()`, at *all levels of noise*. `0` means noiseless, `1` means normal level of noise, and the other numbers are the indices of the noise level.
    - `figs` contains the mitigation results for this specific replication
    - `histograms` contains, as pickle objects, the dictionary to be passed in `plot_histogram` function (Qiskit function)
    - `mitigation_results` contains the extrapolated probability of a specific outcome (float saved as pickle object)
    - `probs` contains the same quantities as `count`, but converted into probabilities
- `stats` gathers the results coming from all `rep_i` directories to extract the statistics

## code files

- `mitigation.ipynb` contains all code needed for the ZNE: fit, extrapolation, creation of the circuits, etc...
- `mitigation.py` python version of `mitigation.ipynb`
- `main.ipynb` uses `mitigation.py` as a black box to run all circuits
- `data_analysis.ipynb` compares the results coming from various noise models, ZNE methods, etc...  



