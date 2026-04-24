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