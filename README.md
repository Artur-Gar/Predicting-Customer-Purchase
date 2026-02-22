# Customer Purchase Prediction

## Description
This repository contains my customer purchase prediction solution for a Kaggle competition run within a Machine Learning course at CentraleSupélec, where it placed 2nd.
It organizes reusable ML code as a Python package under `src/purchase_prediction`, including data loading, cross-validation utilities, iterative KNN imputation, and neural network helpers.  
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
- `notebooks/`: exploratory and training workflow notebook.
- `data/raw/`: input datasets used by notebook.

## Report
Final project report for the Kaggle purchase prediction competition.

[purchase_prediction_kaggle_report.pdf](docs/purchase_prediction_kaggle_report.pdf)

