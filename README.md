# Customer Purchase Prediction

## Description
This repository contains kaggle competition project for customer purchase prediction.  
It organizes reusable machine learning code as a Python package under `src/purchase_prediction`, including data loading, cross-validation utilities, iterative KNN imputation, and neural network helpers.  
The main experimentation workflow is in `notebooks/Purchase_prediction.ipynb`.  

## Setup
```bash
poetry install
poetry shell
```

## Usage
Notebook workflow:
- `notebooks/Purchase_prediction.ipynb`

## Structure
- `src/purchase_prediction/`: importable project modules (data loading, CV, imputation, neural network, pipeline helpers).
- `scripts/`: runnable entrypoints that call package code.
- `notebooks/`: exploratory and training workflow notebook.
- `data/raw/`: input datasets used by scripts and notebook.

## Notes
- `scripts/train.py` expects `data/raw/train_dataset.csv` and `data/raw/test_dataset.csv` to be present.