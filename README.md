# Community Energy Sharing Optimization

This repository contains code and models for a bachelor thesis focused on optimizing energy redistribution in a community energy sharing setting. The main objective is to minimize total unallocated energy (unused producer supply and unmet consumer demand) across a month of 15-minute intervals.

## Project Structure

- **Analysis/**  
  Data exploration and analysis notebooks.
- **Models/**  
  Core optimization models and utilities:
  - `dynamic_weights.py`, `static_weights.py`: Weight optimization models.
  - `heuristics.py`: Heuristic algorithms for preference optimization.
  - `experiments.py`, `testing.ipynb`: Experimentation and testing scripts.
  - `utils.py`: Help functions.
  - `whole_model.py`: Main model combining all components. Nonconvex formulation.
- **Outputs/**  
  Result files, plots, and experiment outputs.

## Problem Overview

The goal is to allocate energy from producers to consumers efficiently, respecting the following constraints:
- Each consumer can receive energy from up to 5 producers.
- Preferences (priorities) are assigned as integers (5 = highest, 1 = lowest, 0 = no connection).
- The allocation must not exceed producer supply or consumer demand.

### Model Requirements

- **Inputs:**  
  - Producer supply and consumer demand time series (15-minute intervals).
  - Preference matrix (consumer-producer priorities).
- **Outputs:**  
  - Feasible weight and priority matrices.
  - Total unredistributed energy.
  - Breakdown of unmet demand and unused supply.

## Optimization Approaches

### Weight Optimization

- **Static and Dynamic Models:**  
  Find optimal energy distribution weights from producers to consumers, given fixed preferences.
- **Approaches:**  
  - High-fidelity (non-convex, slow, accurate).
  - Simplified convex (fast, approximate).

### Preference Optimization

- **Heuristic Algorithms:**  
  - Genetic Algorithm (GA)
  - Simulated Annealing (SA)
  - Local Search (Hill Climbing)
- These methods search for consumer preference structures that minimize unallocated energy.

## Usage

1. **Data Preparation:**  
   Place processed data in the `data/` directory.
2. **Run Models:**  
   Use scripts in `Models/` to perform optimization. Example:
   ```sh
   python Models/dynamic_weights.py
    ```