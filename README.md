# SAGE: Structure-Aware Genetic Compiler Flag Optimisation

SAGE is a system for automatically optimising compiler flags using a genetic algorithm. It targets C benchmarks (e.g. PolyBench) and searches for flag combinations that outperform the standard `-O3` optimisation level.

---

## Features

- Genetic algorithm-based compiler flag optimisation  
- Automatic baseline comparison against `-O3`  
- Support for PolyBench/C benchmarks  
- Configurable via `config.json` or command-line arguments  
- Hyperparameter tuning using OpenTuner  
- Reproducible and extensible design  

---

## Prerequisites

Ensure the following are installed:

- Python 3.11+
- GCC 15.2.0 (or compatible)
- OpenTuner:
  ```bash
  pip install opentuner

---

## Configuration

All GA settings are specified in `config.json` in the 
project root. Edit this file to change the target benchmark, 
hyperparameters, or random seed without modifying source code.

```json
{
  "benchmark": "algorithm/benchmarks/matmul.cpp",
  "compiler": "g++-15",
  "population_size": 64,
  "crossover_type": "one_point",
  "crossover_rate": 0.471,
  "mutation_type": "gauss_by_center",
  "mutation_rate": 0.187,
  "selection_type": "ranking",
  "elitism_ratio": 0.300,
  "parents_portion": 0.553,
  "max_generations": 33,
  "max_no_improvement": 25,
  "num_runs": 5,
  "random_seed": 42
}
```

---

## Running SAGE

1. Clone the repository:

```bash
git clone https://github.com/yarakmk/GeneticAlgorithmProject.git
cd GeneticAlgorithmProject
```

2. Set the target benchmark in `config.json`.

3. Set the compiler commad in `config.json`.

4. Run the optimiser:

```bash
python3 main.py
```

SAGE will print per-generation progress including best 
fitness, average fitness, and elapsed time, and output 
the best flag configuration discovered on completion.

---

## Hyperparameter Tuning

To tune the GA hyperparameters for your benchmark and 
hardware using OpenTuner:

```bash
python3 hyperparameter_tuner.py --test-limit 52
```

Results are saved to `opentuner.db/<ip-address>.db`. 
The best configuration can then be manually copied 
into `config.json`. Note that `opentuner.db` and 
`opentuner.log` are runtime-generated and excluded 
from version control.

---

## Running Tests

Run the full test suite using pytest:

```bash
pytest tests/
```

To include a coverage report:

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Reproducibility

A fixed random seed is set in `config.json` under 
`"random_seed"`. This seeds both Python's `random` 
module and `numpy` at startup. Note that the primary 
experimental results in the accompanying report were 
produced prior to the introduction of the fixed seed 
and may differ slightly across runs due to stochastic 
variation in the GA.

---
## Further Details
For more details on how to configure SAGE, please refer to the User Guide in Appendix B in the report.

## Platform

- OS: macOS 14.0 (Sonoma)
- Architecture: ARM64 (Apple M2)
- GCC: 15.2.0
- Python: 3.12
