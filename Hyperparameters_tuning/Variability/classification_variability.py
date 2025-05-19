# -*- coding: utf-8 -*-
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Initialize the quantum device
dev = qml.device("default.qubit", wires=1)

# Load data
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_train_loader.pth")

# Configuration settings
depth = 10
batch_size = 20
nb_epoch = 30
num_runs = 10  # Number of runs to measure variability

# Determine the input size
for x_train, y_train in train_loader:
    input_size = x_train.shape[1]
    break

# Loss and classification functions
def L2(yh, gt):
    return (yh - gt) ** 2

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

# Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = f"classification_variability_{timestamp}.log"

# Function to run the training and testing process
def run_training(log_file, run):
    # Initialize parameters as a single tensor
    nb_params = depth * 3 * input_size
    params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.Adam([params], lr=0.00005)
    
    test_acc_hist = []
    epoch, cnt = 0, 0

    with open(log_file, "a") as f:
        f.write(f"Starting training loop for run {run+1}\n")
        while epoch < nb_epoch:
            epoch_loss = 0.0
            for x_train, y_train in train_loader:
                opt.zero_grad()
                res = gamma(params, x_train, depth)
                scal_res = res.detach().numpy()
                cl = classif(scal_res)
                loss = L2(res, y_train)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                
                cnt += 1
                if cnt == batch_size:
                    # Evaluate on test set
                    test_acc, test_count, test_stop = 0, 0, 100

                    for x_test, y_test in test_loader:
                        res_test = gamma(params, x_test, depth)
                        if classif(res_test.detach().numpy()) == y_test.numpy()[0][0]:
                            test_acc += 1
                        test_count += 1
                        if test_count == test_stop:
                            break
                    test_acc_fin = test_acc / test_stop
                    test_acc_hist.append(test_acc_fin)
                    
                    cnt = 0
                    
            avg_loss = epoch_loss / len(train_loader)
            f.write(f"Epoch {epoch+1}/{nb_epoch}, Loss: {avg_loss:.6f}\n")
            epoch += 1

    return np.mean(test_acc_hist)

# Run multiple training processes to measure variability
test_acc_results = []
with open(log_file, "w") as f:
    for run in range(num_runs):
        test_acc = run_training(log_file, run)
        test_acc_results.append(test_acc)
        f.write(f"Run {run+1} Test Accuracy: {test_acc:.6f}\n")

# Calculate White Noise (Standard Deviation of test accuracies)
white_noise = np.std(test_acc_results)
mean_acc = np.mean(test_acc_results)

with open(log_file, "a") as f:
    f.write(f"White Noise (Variability of Test Accuracy): {white_noise:.6f}\n")
    f.write(f"Mean Test Accuracy: {mean_acc:.6f}\n")

# Display the White Noise value
print(f"White Noise (Variability of Test Accuracy): {white_noise:.6f}")
print(f"Mean Test Accuracy: {mean_acc:.6f}")

# Plot the distribution of test accuracies
plt.hist(test_acc_results, bins=10, alpha=0.75)
plt.title('Distribution of Test Accuracies')
plt.xlabel('Test Accuracy')
plt.ylabel('Frequency')
plt.axvline(mean_acc, color='r', linestyle='dashed', linewidth=1, label=f'Mean = {mean_acc:.3f}')
plt.axvline(mean_acc - white_noise, color='g', linestyle='dashed', linewidth=1, label=f'-1 SD = {mean_acc - white_noise:.3f}')
plt.axvline(mean_acc + white_noise, color='g', linestyle='dashed', linewidth=1, label=f'+1 SD = {mean_acc + white_noise:.3f}')
plt.legend(loc='upper right')

# Add text box with mean and variability
textstr = f'Mean: {mean_acc:.3f}\nVariability: {white_noise:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.savefig(f"Test_Accuracies_Variability_{timestamp}.png", dpi=300)
plt.show()
