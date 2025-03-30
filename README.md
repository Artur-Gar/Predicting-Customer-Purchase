# 🛍️ Customer Purchase Prediction

This repository contains a full machine learning pipeline for predicting customer purchases using classical ML models, a custom neural network and iterative imputation strategies for missing values.

---

## 📁 Project Structure

<pre>
  . 
  ├── Purchase_prediction.ipynb      # Main notebook with training and evaluation pipeline 
  ├── data/ 
  │ ├── train_dataset.csv            # Training data 
  │ └── test_dataset.csv             # Test data 
  ├── custom_modules/ 
  │ ├── __init__.py 
  │ ├── custom_cv_functions.py       # Cross-validation logic with threshold optimization 
  │ ├── custom_data_imputing.py      # Iterative KNN imputation for missing values 
  │ └── custom_neural_network.py     # Torch-based binary classification network with CV 
</pre>

---

## 🚀 Features

- **Iterative KNN Imputation** for missing data using both regression and classification
- **PyTorch-based Neural Network** for binary classification with cross-validation tuning
- **Stratified Cross-Validation** with threshold tuning via ROC curve
- Modularized Python code for clean experimentation

---

## 📊 Data

The dataset files are located in the `data/` folder:

- `train_dataset.csv`: labeled training data
- `test_dataset.csv`: unlabeled test data for final predictions

---

## 🤖 Custom Modules

All reusable logic is in the `custom_modules/` folder, including:

- `custom_cv_functions.py`: Cross-validation with threshold optimization
- `custom_data_imputing.py`: Iterative KNN imputer class
- `custom_neural_network.py`: Binary classification model in PyTorch and it's tuning with cross-validation

---

## 📝 Author

**Artur Garipov**  
[LinkedIn](https://www.linkedin.com/in/artur-garipov-36037a319) | [GitHub](https://github.com/Artur-Gar)
