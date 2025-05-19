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

# Create the directory for saving outputs
output_dir = "Globale_optimisation_Classification_V3"
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
test_loader = torch.load("loaders/qml_essai_train_loader.pth")

# Determine the input size
for x_train, y_train in train_loader:
    input_size = x_train.shape[1]
    break

# Define the hyperparameter search space
space = [
    Categorical([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='depth'),
    Categorical([0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005, 0.0000005], name='learning_rate'),
    Categorical(['L1', 'L2', 'Huber'], name='loss_function'),
    Categorical(['SGD', 'Adam', 'Adagrad', 'Adadelta'], name='optimizer'),
    Categorical(['none', '-pi_to_pi', '0_to_2pi'], name='normalization')
]

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

# Normalization function
def normalize_data(x, normalization):
    if normalization == '-pi_to_pi':
        return (x - x.min()) / (x.max() - x.min()) * 2 * np.pi - np.pi
    elif normalization == '0_to_2pi':
        return (x - x.min()) / (x.max() - x.min()) * 2 * np.pi
    return x

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
@use_named_args(space)
def objective(**params):
    depth = params['depth']
    learning_rate = params['learning_rate']
    loss_function = params['loss_function']
    optimizer_name = params['optimizer']
    normalization = params['normalization']
    nb_epochs = 20  # Modify the number of epochs to 20
    
    with open(log_file, 'a') as f:
        f.write(f"\nTesting configuration: depth={depth}, learning_rate={learning_rate}, "
                f"loss_function={loss_function}, optimizer={optimizer_name}, normalization={normalization}\n")
    
    print(f"\nTesting configuration: depth={depth}, learning_rate={learning_rate}, "
          f"loss_function={loss_function}, optimizer={optimizer_name}, normalization={normalization}")
    
    # Initialize parameters as a single tensor in the range [-pi, pi]
    nb_params = depth * 3 * input_size
    q_params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
    
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
        csvwriter.writerow(['Epoch', 'Loss', 'Accuracy', 'Test Accuracy'])
    
        for epoch in range(nb_epochs):
            epoch_loss, epoch_acc = 0.0, 0
            for x_train, y_train in train_loader:
                opt.zero_grad()
                
                # Normalize the data
                x_train = normalize_data(x_train, normalization)
                
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
            
            # Evaluate on test set
            test_acc, test_count = 0, 0
            for x_test, y_test in test_loader:
                res_test = gamma(q_params, x_test, depth)
                if classif(res_test.detach().numpy()) == y_test.numpy()[0][0]:
                    test_acc += 1
                test_count += 1
            avg_test_acc = test_acc / test_count
            
            print(f"Epoch {epoch+1}/{nb_epochs} - avg_loss={avg_loss:.6f}, avg_acc={avg_acc:.6f}, avg_test_acc={avg_test_acc:.6f}")
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1}/{nb_epochs} - avg_loss={avg_loss:.6f}, avg_acc={avg_acc:.6f}, avg_test_acc={avg_test_acc:.6f}\n")
            
            csvwriter.writerow([epoch + 1, avg_loss, avg_acc, avg_test_acc])
    
    return avg_loss

# Execute Bayesian optimization
print("Starting Bayesian optimization...")
with open(log_file, 'a') as f:
    f.write("Starting Bayesian optimization...\n")

res_gp = gp_minimize(objective, space, acq_func='LCB', kappa=4.0, n_calls=50, random_state=0)

print("Bayesian optimization completed.")
with open(log_file, 'a') as f:
    f.write("Bayesian optimization completed.\n")

# Display optimization results
print(f"\nMeilleur score: {res_gp.fun}")
print(f"Meilleurs hyperparamètres: {res_gp.x}")

with open(log_file, 'a') as f:
    f.write(f"\nMeilleur score: {res_gp.fun}\n")
    f.write(f"Meilleurs hyperparamètres: {res_gp.x}\n")

# Plot convergence
plot_convergence(res_gp)
plt.savefig(figure_file)
plt.show()

# Evaluate the best hyperparameters found
best_depth = res_gp.x[0]
best_learning_rate = res_gp.x[1]
best_loss_function = res_gp.x[2]
best_optimizer = res_gp.x[3]
best_normalization = res_gp.x[4]

print(f"\nBest Depth: {best_depth}")
print(f"Best Learning Rate: {best_learning_rate}")
print(f"Best Loss Function: {best_loss_function}")
print(f"Best Optimizer: {best_optimizer}")
print(f"Best Normalization: {best_normalization}")

with open(log_file, 'a') as f:
    f.write(f"\nBest Depth: {best_depth}\n")
    f.write(f"Best Learning Rate: {best_learning_rate}\n")
    f.write(f"Best Loss Function: {best_loss_function}\n")
    f.write(f"Best Optimizer: {best_optimizer}\n")
    f.write(f"Best Normalization: {best_normalization}\n")
