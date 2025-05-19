# -*- coding: utf-8 -*-
import sys
import subprocess
import site

# Add the user site-packages directory to sys.path
site.addsitedir(site.getusersitepackages())

# Function to install a package using pip
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])

# Check if scikit-optimize is installed, and install if necessary
try:
    import skopt
except ImportError:
    print("scikit-optimize is not installed. Installing...")
    install_package("scikit-optimize")
    import skopt

import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os
import csv
from scipy.interpolate import UnivariateSpline
import time
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

# Import BOParOpt
from bopar import BOParOpt, ucb, thresh_stop_crit

# Create the directory for saving outputs
output_dir = "Globale_optimisation_BOPAR_Classification_V2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# File paths for logging and raw outputs
log_file = os.path.join(output_dir, "optimization_log.txt")
csv_file = os.path.join(output_dir, "raw_outputs.csv")
figure_file = os.path.join(output_dir, "convergence_plot.png")

# Initialize the quantum device
dev = qml.device("default.qubit", wires=1)

# Load data
prefix = "qml"
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_test_loader.pth")

# Determine the input size
for x_train, y_train in train_loader:
    input_size = x_train.shape[1]
    break

# Define the hyperparameter search space
categorical_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
categorical_learning_rate = [0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005, 0.0000005]
categorical_loss_function = ['L1', 'L2', 'Huber']
categorical_optimizer = ['SGD', 'Adam', 'Adagrad', 'Adadelta']

# Convert categorical parameters to indices
def encode_categorical_params(params):
    return [
        categorical_depth.index(params[0]),
        categorical_learning_rate.index(params[1]),
        categorical_loss_function.index(params[2]),
        categorical_optimizer.index(params[3])
    ]

def decode_categorical_params(params):
    return [
        categorical_depth[int(params[0])],
        categorical_learning_rate[int(params[1])],
        categorical_loss_function[int(params[2])],
        categorical_optimizer[int(params[3])]
    ]

# Define the hyperparameter search space for BOParOpt
pbounds = [(0, len(categorical_depth) - 1), (0, len(categorical_learning_rate) - 1),
           (0, len(categorical_loss_function) - 1), (0, len(categorical_optimizer) - 1)]

# Define loss functions
def L1(yh, gt):
    return torch.abs(yh - gt)

def L2(yh, gt):
    return (yh - gt) ** 2

def Huber(yh, gt, delta=1.0):
    diff = yh - gt
    abs_diff = torch.abs(diff)
    return torch.where(abs_diff < delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))

# Classification function
def classif(yh):
    if yh < -0.33:
        return -1.0
    if yh > 0.33:
        return 1.0
    return 0.0

# Quantum node definition
@qml.qnode(dev, interface="torch")
def gamma(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RX(params[j * 3 + i * input_size * 3], wires=0)
            qml.RY((params[j * 3 + 1 + i * input_size * 3]) * (x[0][j].item()), wires=0)
            qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

# Define the objective function for Bayesian optimization
def objective(params):
    params = decode_categorical_params(params)
    depth = params[0]
    learning_rate = params[1]
    loss_function = params[2]
    optimizer_name = params[3]
    nb_epochs = 20  # Modify the number of epochs to 20
    
    with open(log_file, 'a') as f:
        f.write(f"\nTesting configuration: depth={depth}, learning_rate={learning_rate}, "
                f"loss_function={loss_function}, optimizer={optimizer_name}\n")
    
    print(f"\nTesting configuration: depth={depth}, learning_rate={learning_rate}, "
          f"loss_function={loss_function}, optimizer={optimizer_name}")
    
    # Initialize parameters as a single tensor in the range [-pi, pi]
    nb_params = depth * 3 * input_size
    q_params = torch.tensor((np.random.rand(nb_params) * 2 * np.pi - np.pi), requires_grad=True, dtype=torch.float64)
    
    if optimizer_name == 'SGD':
        opt = torch.optim.SGD([q_params], lr=learning_rate)
    elif optimizer_name == 'Adam':
        opt = torch.optim.Adam([q_params], lr=learning_rate)
    elif optimizer_name == 'Adagrad':
        opt = torch.optim.Adagrad([q_params], lr=learning_rate)
    elif optimizer_name == 'Adadelta':
        opt = torch.optim.Adadelta([q_params], lr=learning_rate)
    
    with open(csv_file, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'Loss', 'Accuracy'])
    
        for epoch in range(nb_epochs):
            epoch_loss, epoch_acc = 0.0, 0
            for x_train, y_train in train_loader:
                opt.zero_grad()
                res = gamma(q_params, x_train, depth)
                
                if loss_function == 'L1':
                    loss = L1(res, y_train)
                elif loss_function == 'L2':
                    loss = L2(res, y_train)
                elif loss_function == 'Huber':
                    loss = Huber(res, y_train)
                
                loss.backward()
                opt.step()
                
                epoch_loss += loss.detach().numpy()[0][0]
                if classif(res.detach().numpy()) == y_train.numpy()[0][0]:
                    epoch_acc += 1
            
            avg_loss = epoch_loss / len(train_loader)
            avg_acc = epoch_acc / len(train_loader)
            
            print(f"Epoch {epoch+1}/{nb_epochs} - avg_loss={avg_loss:.6f}, avg_acc={avg_acc:.6f}")
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1}/{nb_epochs} - avg_loss={avg_loss:.6f}, avg_acc={avg_acc:.6f}\n")
            
            csvwriter.writerow([epoch + 1, avg_loss, avg_acc])
    
    return avg_loss

# Initialize BOParOpt
bo = BOParOpt(f=objective, pbounds=pbounds)

# Execute Bayesian optimization with parallelism
print("Starting Bayesian optimization with parallelism...")
with open(log_file, 'a') as f:
    f.write("Starting Bayesian optimization with parallelism...\n")

# Define the acquisition function and the stopping criterion
acq_func = ucb
crit = thresh_stop_crit

# Custom function to generate initial samples correctly
def generate_initial_samples(pbounds, n_samples):
    samples = []
    for bound in pbounds:
        samples.append(np.random.uniform(bound[0], bound[1], n_samples))
    return np.array(samples).T

# Generate initial samples
init_samples = generate_initial_samples(pbounds, 5)
bo.samples_coords = init_samples
bo.samples_eval = np.array([objective(sample) for sample in init_samples])

# Correction dans batch_minimize pour générer les échantillons correctement
def batch_minimize(self, batch_size, init_points, crit, acq_func):
    # Génération correcte des échantillons initiaux
    self.samples_coords = np.array([np.random.uniform(bound[0], bound[1], init_points) for bound in self.pbounds]).T
    self.samples_eval = np.array([self.f(sample) for sample in self.samples_coords])

    # Poursuite du processus d'optimisation
    # Le reste de votre code ici

bo.batch_minimize(batch_size=10, init_points=5, crit=crit, acq_func=acq_func)

print("Bayesian optimization with parallelism completed.")
with open(log_file, 'a') as f:
    f.write("Bayesian optimization with parallelism completed.\n")

# Plot convergence
bo.plot()
plt.savefig(figure_file)
plt.show()

# Print and log the best hyperparameters found
best_params = decode_categorical_params(bo.best_coords)
best_depth = best_params[0]
best_learning_rate = best_params[1]
best_loss_function = best_params[2]
best_optimizer = best_params[3]

print(f"\nBest Depth: {best_depth}")
print(f"Best Learning Rate: {best_learning_rate}")
print(f"Best Loss Function: {best_loss_function}")
print(f"Best Optimizer: {best_optimizer}")

with open(log_file, 'a') as f:
    f.write(f"\nBest Depth: {best_depth}\n")
    f.write(f"Best Learning Rate: {best_learning_rate}\n")
    f.write(f"Best Loss Function: {best_loss_function}\n")
    f.write(f"Best Optimizer: {best_optimizer}\n")
