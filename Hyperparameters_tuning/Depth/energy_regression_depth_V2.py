# -*- coding: utf-8 -*-
import os
import pennylane as qml
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize the quantum device
dev = qml.device("default.qubit", wires=1)

# Create directory for saving results
output_dir = "regression_depth"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data
print("Loading data...")
data = pd.read_csv('Dataregression/electrons.csv', header=None)

# Normalize the first three features between 0 and 2pi
print("Normalizing the first three features between 0 and 2pi...")
for col in data.columns[:3]:
    data[col] = 2 * np.pi * (data[col] - data[col].min()) / (data[col].max() - data[col].min())

# Display the first few rows to verify normalization
print("Data after normalization:")
print(data.head())

# Extract features and energies
features = data.iloc[:, :-1].values
energies = data.iloc[:, -1].values

# Convert data to PyTorch tensors
print("Converting data to PyTorch tensors...")
features = torch.tensor(features, dtype=torch.float64)
energies = torch.tensor(energies, dtype=torch.float64).view(-1, 1)

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(features, energies, test_size=0.2, random_state=42)

# Configuration settings
nb_epoch = 100
input_size = X_train.shape[1]
lr = 0.005  # Updated learning rate

# Define the dataset and dataloader
print("Creating DataLoader...")
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Loss function (Huber loss)
def huber_loss(y_pred, y_true, delta=1.0):
    loss = torch.where(torch.abs(y_pred - y_true) < delta,
                       0.5 * (y_pred - y_true) ** 2,
                       delta * (torch.abs(y_pred - y_true) - 0.5 * delta))
    return loss.mean()

# Updated accuracy function for regression
def acc(pred, gt):
    pred = pred.detach().numpy()
    gt = gt.numpy()
    l1 = np.abs(gt - pred)
    epsilon = 1e-10  # Petite valeur pour éviter la division par zéro
    rel_err = l1 / (gt + epsilon)
    err_acc = np.maximum(1 - rel_err, 0.0)
    return err_acc.mean()

# Quantum node definition
@qml.qnode(dev, interface="torch")
def gamma(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RX(params[j * 3 + i * input_size * 3], wires=0)
            qml.RY(params[j * 3 + 1 + i * input_size * 3] * x[j], wires=0)
            qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

log_file = os.path.join(output_dir, "train_qml_energy_regression.log")
with open(log_file, "w") as f:
    # Store metrics for plotting
    depths = range(1, 16)
    trainability_list = []
    test_acc_list = []
    loss_list = []
    dispersion_list = []
    execution_time_list = []
    raw_output = []

    # Function to train and evaluate the model for a given depth
    def train_and_evaluate(depth):
        print(f"Training with depth: {depth}")
        f.write(f"Training with depth: {depth}\n")
    
        # Initialize parameters as a single tensor
        nb_params = depth * 3 * input_size
        params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
        opt = torch.optim.SGD([params], lr=lr)

        # Training loop
        print("Starting training loop...")
        f.write("Starting training loop...\n")
        loss_hist = []
        train_acc_hist = []
        test_acc_hist = []

        start_time_process = time.time()
        for epoch in range(nb_epoch):
            print(f"Epoch {epoch+1}/{nb_epoch}")
            f.write(f"Epoch {epoch+1}/{nb_epoch}\n")
            epoch_loss = 0.0
            train_acc = 0.0
            for batch_idx, (x_train, y_train) in enumerate(train_loader):
                if batch_idx % 10 == 0:
                    print(f"Processing example {batch_idx+1}/{len(train_loader)}")
                    f.write(f"Processing example {batch_idx+1}/{len(train_loader)}\n")
                opt.zero_grad()
                res = gamma(params, x_train[0], depth)
                loss = huber_loss(res, y_train)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                train_acc += acc(res, y_train)

            avg_loss = epoch_loss / len(train_loader)
            avg_train_acc = train_acc / len(train_loader)
            loss_hist.append(avg_loss)
            train_acc_hist.append(avg_train_acc)

            # Evaluate on test set
            test_acc = 0.0
            for x_test, y_test in test_loader:
                test_res = gamma(params, x_test[0], depth)
                test_acc += acc(test_res, y_test)
            avg_test_acc = test_acc / len(test_loader)
            test_acc_hist.append(avg_test_acc)

            print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.6f}, Train Accuracy: {avg_train_acc:.6f}, Test Accuracy: {avg_test_acc:.6f}")
            f.write(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.6f}, Train Accuracy: {avg_train_acc:.6f}, Test Accuracy: {avg_test_acc:.6f}\n")

        end_time_process = time.time()
        execution_time = end_time_process - start_time_process

        # Prediction function
        def predict(params, x):
            with torch.no_grad():
                return gamma(params, x, depth).detach().numpy()

        # Evaluate the model on the test set
        print("Evaluating the model on the test set...")
        f.write("Evaluating the model on the test set...\n")
        y_pred = []
        y_true = []

        for x_test, y_test in test_loader:
            for x, y in zip(x_test, y_test):
                y_pred.append(predict(params, x))
                y_true.append(y.item())

        # Convert predictions and true values to numpy arrays
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        dispersion = np.abs(y_true - y_pred)  # Calculate dispersion for each point

        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"R^2 Score: {r2:.6f}")
        f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
        f.write(f"R^2 Score: {r2:.6f}\n")

        # Save raw output data
        raw_output.append({
            'depth': depth,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'execution_time': execution_time,
            'trainability': np.trapz(loss_hist),
            'avg_test_acc': avg_test_acc,
            'avg_loss': avg_loss,
            'avg_dispersion': dispersion.mean()
        })

        # Create figures
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Plot training loss and accuracy
        ax[0].plot(loss_hist, label='Training Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].set_title('Training Loss')

        ax[1].plot(train_acc_hist, label='Training Accuracy')
        ax[1].plot(test_acc_hist, label='Test Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        ax[1].set_title('Accuracy')

        plt.suptitle(f'Energy regression electrons (Depth={depth})')
        plt.savefig(os.path.join(output_dir, f"Energy_regression_electrons_depth_{depth}_loss_acc.png"), dpi=300)
        plt.show()

        # Create a figure with True vs Predicted and Mean Dispersion
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Plot true vs predicted values
        ax[0].scatter(y_true, y_pred, alpha=0.6)
        ax[0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red')
        ax[0].set_xlabel('True Values')
        ax[0].set_ylabel('Predicted Values')
        ax[0].set_title('True vs Predicted Values')

        # Plot mean dispersion of predicted values around true values
        ax[1].scatter(y_true, dispersion, alpha=0.6)
        ax[1].set_xlabel('True Values')
        ax[1].set_ylabel('Dispersion')
        ax[1].set_title('Mean Dispersion of Predicted Values')

        plt.suptitle(f'Energy regression electrons (Depth={depth})')
        plt.savefig(os.path.join(output_dir, f"Energy_regression_electrons_depth_{depth}_true_pred_dispersion.png"), dpi=300)
        plt.show()

        # Calculate trainability as the area under the loss curve
        trainability = np.trapz(loss_hist)
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Trainability: {trainability:.6f}")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n")
        f.write(f"Trainability: {trainability:.6f}\n")

        # Store metrics for plotting
        trainability_list.append(trainability)
        test_acc_list.append(avg_test_acc)
        loss_list.append(avg_loss)
        dispersion_list.append(dispersion.mean())
        execution_time_list.append(execution_time)

    # Train and evaluate the model for each depth
    for depth in depths:
        train_and_evaluate(depth)
    
    f.write("Execution completed\n")

    # Save raw output to CSV
    raw_output_df = pd.DataFrame(raw_output)
    raw_output_df.to_csv(os.path.join(output_dir, "raw_output.csv"), index=False)

    # Plot metrics vs depth
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(depths, trainability_list, marker='o')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Trainability')
    ax.set_title('Trainability vs Depth')
    plt.savefig(os.path.join(output_dir, "Trainability_vs_Depth.png"), dpi=300)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(depths, test_acc_list, marker='o')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Test Accuracy')
    ax.setTitle('Test Accuracy vs Depth')
    plt.savefig(os.path.join(output_dir, "Test_Accuracy_vs_Depth.png"), dpi=300)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(depths, loss_list, marker='o')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Depth')
    plt.savefig(os.path.join(output_dir, "Loss_vs_Depth.png"), dpi=300)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(depths, dispersion_list, marker='o')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Dispersion')
    ax.set_title('Dispersion vs Depth')
    plt.savefig(os.path.join(output_dir, "Dispersion_vs_Depth.png"), dpi=300)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(depths, execution_time_list, marker='o')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Execution Time vs Depth')
    plt.savefig(os.path.join(output_dir, "Execution_Time_vs_Depth.png"), dpi=300)
    plt.show()
