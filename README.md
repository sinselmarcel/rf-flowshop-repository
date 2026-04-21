# rf-flowshop-repository

Simulation-based comparison of heuristic and Random Forest dispatching in a dynamic flow shop with Workload Control.

## Overview

This repository contains the Python code used to simulate and evaluate dispatching policies in a dynamic three-stage flow shop with Workload Control (WLC). The project compares three classical heuristics against a machine-learning-based dispatcher:

- SPT
- EDD
- ATC
- ML (Random Forest-based dispatcher)

The simulation environment is used for two purposes:

1. To generate training data from heuristic runs
2. To evaluate the trained ML dispatcher against the heuristic benchmarks under identical scenario conditions

## Research Context

The project studies scheduling decisions in a dynamic production system with:

- stochastic job arrivals
- machine-specific stochastic processing times
- machine breakdowns
- due date variation
- workload-controlled order release

The main objective is to analyze trade-offs between service level and logistic performance, especially with respect to:

- on-time rate
- tardiness tail performance
- cycle time
- work-in-process
- throughput

## Repository Contents

### Core files

- `sim_core.py`  
  Main simulation logic for the three-stage flow shop, dispatching, WLC mechanism, disturbances, logging, and KPI generation.

- `run_pilot.py`  
  Runs a single simulation instance using a configuration file and command-line overrides.

- `policy_ml.py`  
  Builds ML candidate features and applies the trained Random Forest models for dispatching decisions.

### Experiment and data pipeline

- `run_grid.py`  
  Runs full experiment grids across policies, due-date scenarios, disruption levels, WLC settings, and seeds.

- `collect_runs.py`  
  Collects run outputs from multiple simulation folders and merges them into aggregated datasets.

- `label_make.py`  
  Generates training labels for ML by combining decision records with realized completion information.

- `train_rct.py`  
  Trains the Random Forest model for remaining cycle time prediction.

- `train_tard.py`  
  Trains the Random Forest model for tardiness prediction.

- `quick_summary.py`  
  Aggregates KPI outputs across runs and produces summary files for evaluation.

### Configuration

- `base.yaml`  
  Main simulation configuration file containing scenario settings, processing times, disturbances, WLC settings, policies, and logging options.

## Requirements

The project is written in Python and uses the following main packages:

- numpy
- pandas
- pyyaml
- simpy
- scikit-learn
- joblib
- pyarrow

## Installation

Install the required Python packages with:

```bash
pip install numpy pandas pyyaml simpy scikit-learn joblib pyarrow
````

A recent Python 3 version is recommended, such as Python 3.10 or newer.

## Seed Split

The project uses separate seed sets for model training and benchmark evaluation.

**Training seeds:**
42, 73, 99, 111, 808, 1337, 2025, 2601, 2718, 31415

**Test seeds:**
222, 333, 444, 555, 666, 777, 888, 999, 1212, 1717

This separation is used to avoid information leakage between model training and evaluation.

## Typical Workflow

A typical workflow consists of the following steps.

### 1. Run training scenarios

```bash
python run_grid.py --cfg base.yaml --policies SPT,EDD,ATC --seeds train
```

### 2. Collect training run outputs

```bash
python collect_runs.py --runs-dir runs
```

### 3. Generate labeled ML data

```bash
python label_make.py
```

### 4. Train the Random Forest models

```bash
python train_rct.py
python train_tard.py
```

### 5. Run evaluation scenarios

```bash
python run_grid.py --cfg base.yaml --seeds test
```

### 6. Generate KPI summaries for evaluation

```bash
python quick_summary.py --runs_dir runs --subset test
```

## Output Structure

Typical output folders used during execution include:

* `runs/`
* `runs_nowlc/`
* `out/`
* `models/`

These folders are usually generated during the simulation and ML workflow and are not required as part of the core repository code.

## Reproducibility

To reproduce results consistently, keep the following fixed:

* the configuration file
* the seed split
* the scenario combinations
* the feature definitions
* the model training procedure
* the evaluation workflow

The repository is intended to document the simulation and evaluation pipeline underlying the comparative analysis of heuristic and ML-based dispatching rules.

## Notes

* The repository currently contains the core scripts and configuration needed for the simulation and ML workflow.
* Large generated outputs, trained models, and local experiment folders are intentionally not included by default.
* The repository is intended to document the code base and experimental pipeline rather than to store raw result data.
