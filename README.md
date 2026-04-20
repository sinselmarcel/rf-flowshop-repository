# rf-flowshop-repository

Simulation-based comparison of heuristic and Random Forest dispatching in a dynamic flow shop with Workload Control.

## Overview

This repository contains the Python code used to simulate and evaluate dispatching policies in a dynamic three-stage flow shop with Workload Control (WLC). The project compares three classical heuristics against a machine-learning-based dispatcher:

- SPT
- EDD
- ATC
- ML (RF) (Random Forest-based dispatcher)

The simulation environment is used for two purposes:

1. to generate training data from heuristic runs
2. to evaluate the trained ML dispatcher against the heuristic benchmarks under identical scenario conditions

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

You can install the dependencies with:

```bash
pip install numpy pandas pyyaml simpy scikit-learn joblib pyarrow


