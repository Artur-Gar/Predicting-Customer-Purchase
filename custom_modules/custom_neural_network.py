import numpy as np
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import itertools
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Set the random seed for reproducibility
random_state = 42
torch.manual_seed(random_state)  # PyTorch CPU random seed
torch.cuda.manual_seed_all(random_state)  # PyTorch GPU random seed
np.random.seed(random_state)  # NumPy random seed
random.seed(random_state)  # Python's random seed

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Define the Neural Network class
class BinaryClassificationNN(nn.Module):
    def __init__(self, input_size, hidden_layers, activation_fn, dropout_rate):
        super(BinaryClassificationNN, self).__init__()
        layers = []
        current_size = input_size
        
        for layer_size in hidden_layers:
            layers.append(nn.Linear(current_size, layer_size))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            current_size = layer_size
        
        layers.append(nn.Linear(current_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Cross-validation function with Stratified K-Fold and optimal threshold
def stratified_cross_validate_nn_with_optimal_threshold(X_train, y_train, param_grid, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    X_train = X_train.values
    y_train = y_train.values

    for params in itertools.product(*param_grid.values()):
        config = dict(zip(param_grid.keys(), params))
        #print(f"Testing config: {config}")
        
        f1_scores = []
        for train_index, val_index in skf.split(X_train, y_train):
            # Split data
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
            
            # Convert to tensors
            X_fold_train_tensor = torch.tensor(X_fold_train, dtype=torch.float32).to(device)
            y_fold_train_tensor = torch.tensor(y_fold_train, dtype=torch.float32).to(device)
            X_fold_val_tensor = torch.tensor(X_fold_val, dtype=torch.float32).to(device)
            y_fold_val_tensor = torch.tensor(y_fold_val, dtype=torch.float32).to(device)
            
            # Create model
            model = BinaryClassificationNN(
                input_size=X_train.shape[1],
                hidden_layers=config["hidden_layers"],
                activation_fn=config["activation_fn"],
                dropout_rate=config["dropout_rate"]
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            #criterion = nn.BCELoss()
            train_dataset = TensorDataset(X_fold_train_tensor, y_fold_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
            
            # Train the model
            model.train()
            for epoch in range(config["epochs"]):
                for batch_X, batch_y in train_loader:
                    # Move data to GPU
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    weights = batch_y.cpu().numpy()
                    weight_for_ones = weights[weights==0].shape[0] / weights[weights==1].shape[0]
                    weight_for_zeros = 1
                    weights[weights==1] = weight_for_ones 
                    weights[weights==0] = weight_for_zeros
                    weights = torch.tensor(weights).to(device)
                    #weights = None

                    # Forward pass
                    optimizer.zero_grad()
                    predictions = model(batch_X).squeeze()  # Get predictions
                    loss = nn.BCELoss(weight=weights)(predictions, batch_y)
                    #loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Validate the model
            model.eval()
            with torch.no_grad():
                y_probs_X_train = model(X_fold_train_tensor).squeeze()
                y_probs_X_test = model(X_fold_val_tensor).squeeze()
                
                # Compute ROC curve and find optimal threshold
                fpr, tpr, thresholds = roc_curve(y_fold_train_tensor.cpu().numpy(), y_probs_X_train.cpu().numpy())
                optimal_threshold = thresholds[np.argmax(tpr - fpr)]
                #print(f"Optimal Threshold: {optimal_threshold:.4f}")
                
                # Apply optimal threshold
                val_predictions = (y_probs_X_test >= optimal_threshold).int()
                macro_f1 = f1_score(y_fold_val_tensor.cpu().numpy(), val_predictions.cpu().numpy(), average="macro")
                f1_scores.append(macro_f1)
        
        avg_f1 = sum(f1_scores) / len(f1_scores)
        #print(f"Avg Macro F1-Score: {avg_f1:.4f},   Config {config} \n")
        results.append((config, avg_f1))
    
    return results


# Final prediction
def nn_predictions(X, y, X_target, best_config):

    # Convert to tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)
    X_target_tensor = torch.tensor(X_target.values, dtype=torch.float32).to(device)
                
    # Create model
    model = BinaryClassificationNN(
        input_size= X.shape[1],
        hidden_layers=best_config["hidden_layers"],
        activation_fn=best_config["activation_fn"],
        dropout_rate=best_config["dropout_rate"]
    ).to(device)
                
    optimizer = optim.Adam(model.parameters(), lr=best_config["learning_rate"], weight_decay=best_config["weight_decay"])
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True)
                
    # Train the model
    model.train()
    for epoch in range(best_config["epochs"]):
        for batch_X, batch_y in train_loader:
            # Move data to GPU
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            weights = batch_y.cpu().numpy()
            weight_for_ones = weights[weights==0].shape[0] / weights[weights==1].shape[0]
            weight_for_zeros = 1
            weights[weights==1] = weight_for_ones 
            weights[weights==0] = weight_for_zeros
            weights = torch.tensor(weights).to(device)
            #weights = None

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()  
            loss = nn.BCELoss(weight=weights)(predictions, batch_y)
            loss.backward()
            optimizer.step()
                
    # Validate the model
    model.eval()
    with torch.no_grad():
            y_probs_X = model(X_tensor).squeeze()
            y_probs_X_target = model(X_target_tensor).squeeze()
                    
            # Compute ROC curve and find optimal threshold
            fpr, tpr, thresholds = roc_curve(y_tensor.cpu().numpy(), y_probs_X.cpu().numpy())
            optimal_threshold = thresholds[np.argmax(tpr - fpr)]
            print(f"Optimal Threshold: {optimal_threshold:.4f}")
                    
            # Apply optimal threshold
            y_pred_target = (y_probs_X_target >= optimal_threshold).int()
    
    return y_pred_target.cpu().numpy()
