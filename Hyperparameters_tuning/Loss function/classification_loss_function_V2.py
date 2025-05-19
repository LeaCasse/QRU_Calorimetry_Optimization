# -*- coding: utf-8 -*-
import pennylane as qml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import sys
import time
import os
import csv

# Initialize the quantum device
dev = qml.device("default.qubit", wires=1)

# Load data
prefix = "qml"
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_train_loader.pth")

# Configuration settings
depth = 10
batch_size = 20
nb_epoch = 30

# Determine the input size
for x_train, y_train in train_loader:
    input_size = x_train.shape[1]
    break

# Create directory for saving results
output_dir = "classification_loss_functions"
os.makedirs(output_dir, exist_ok=True)

# Loss and classification functions
def L2(yh, gt):
    return (yh - gt) ** 2

def L1(yh, gt):
    return torch.abs(yh - gt)

def Huber(yh, gt, delta=1.0):
    abs_diff = torch.abs(yh - gt)
    return torch.where(abs_diff < delta, 0.5 * abs_diff ** 2, delta * (abs_diff - 0.5 * delta))

def classif(yh):
    if yh < -0.33:
        return -1.0
    if yh > 0.33:
        return 1.0
    return 0.0

# Function to compute moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to compute the mean of the last window_size values
def final_moving_average(data, final_window_size):
    return np.mean(data[-final_window_size:])

# Loss functions to test
loss_functions = {'L2': L2, 'L1': L1, 'Huber': Huber}
trainability_list = []
execution_times = []
final_loss_list = []
final_train_acc_list = []
final_test_acc_list = []
param_means = []
param_stds = []

# Prepare CSV file for raw output
csv_file_path = os.path.join(output_dir, "raw_output.csv")
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Loss Function", "Epoch", "Batch Count", "Avg Loss", "Avg Accuracy", "Test Accuracy"])

    for loss_name, loss_fn in loss_functions.items():
        # Initialize parameters as a single tensor
        nb_params = depth * 3 * input_size
        params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
        opt = torch.optim.Adam([params], lr=0.00005)

        # Quantum node definition
        @qml.qnode(dev, interface="torch")
        def gamma(params, x, depth):
            for i in range(depth):
                for j in range(input_size):
                    qml.RX(params[j * 3  + i * input_size * 3], wires=0)
                    qml.RY((params[j * 3  + 1 + i * input_size * 3]) * (x[0][j].item()), wires=0)
                    qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
            return qml.expval(qml.PauliZ(0))

        loss_hist, acc_hist, test_acc_hist = [], [], []
        epoch, bcnt, bloss, bacc, cnt = 0, 0, 0.0, 0, 0

        start_time = time.time()
        log_file_path = os.path.join(output_dir, f"train_qml_loss_function_{loss_name}.txt")
        with open(log_file_path, "w") as f:
            while epoch < nb_epoch:
                for x_train, y_train in train_loader:
                    opt.zero_grad()
                    res = gamma(params, x_train, depth)
                    scal_res = res.detach().numpy()
                    cl = classif(scal_res)
                    cl_gt = y_train.numpy()[0][0]
                    loss = loss_fn(res, y_train)
                    loss.backward()
                    opt.step()
                    c = loss.detach().numpy()[0][0]
                    bloss += c

                    if cl == cl_gt:
                        bacc += 1

                    cnt += 1

                    if cnt == batch_size:
                        avg_loss = bloss / batch_size
                        avg_acc = bacc / batch_size
                        loss_hist.append(avg_loss)
                        acc_hist.append(avg_acc)

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

                        bloss, bacc, cnt = 0.0, 0, 0

                        msg = f"epoch={epoch} bcnt={bcnt} loss={avg_loss:.6f} acc={avg_acc:.6f} test acc={test_acc_fin:.6f}"
                        print(msg)
                        f.write(msg + "\n")
                        csv_writer.writerow([loss_name, epoch, bcnt, avg_loss, avg_acc, test_acc_fin])

                        bcnt += 1

                epoch += 1

        end_time = time.time()
        execution_times.append(end_time - start_time)

        # Compute moving averages
        window_size = 1000
        final_window_size = 10
        loss_hist_ma = moving_average(loss_hist, window_size)
        acc_hist_ma = moving_average(acc_hist, window_size)
        test_acc_hist_ma = moving_average(test_acc_hist, window_size)

        # Compute final moving averages
        final_loss_ma = final_moving_average(loss_hist_ma, final_window_size)
        final_train_acc_ma = final_moving_average(acc_hist_ma, final_window_size)
        final_test_acc_ma = final_moving_average(test_acc_hist_ma, final_window_size)

        # Compute trainability (area under the loss curve)
        batches_per_epoch = len(loss_hist) // nb_epoch
        epochs = np.arange(len(loss_hist_ma)) / batches_per_epoch
        trainability = np.trapz(loss_hist_ma, epochs)

        # Collect results
        trainability_list.append(trainability)
        final_loss_list.append(final_loss_ma)
        final_train_acc_list.append(final_train_acc_ma)
        final_test_acc_list.append(final_test_acc_ma)
        param_means.append(params.detach().numpy().mean())
        param_stds.append(params.detach().numpy().std())

        # Plot the distribution of all parameters
        plt.hist(params.detach().cpu().numpy(), bins=50, alpha=0.75)
        plt.title(f'Distribution of parameters values (loss function = {loss_name})')
        plt.xlabel('Parameter value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f"Distribution_of_parameters_values_{loss_name}.png"), dpi=300)
        plt.show()

        # Plot loss and accuracy
        plt.figure(figsize=(14, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss_hist_ma, label='Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, acc_hist_ma, label='Train accuracy')
        plt.plot(epochs, test_acc_hist_ma, label='Test accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.suptitle(f'Loss and Accuracy (loss function = {loss_name})')
        plt.savefig(os.path.join(output_dir, f"Loss_Accuracy_{loss_name}.png"), dpi=300)
        plt.show()

# Plot results
plt.figure(figsize=(12, 10))

# Trainability vs loss function
plt.subplot(3, 2, 1)
plt.plot(list(loss_functions.keys()), trainability_list, marker='o')
plt.title('Trainability vs Loss Function')
plt.xlabel('Loss Function')
plt.ylabel('Trainability')

# Execution time vs loss function
plt.subplot(3, 2, 2)
plt.plot(list(loss_functions.keys()), execution_times, marker='o')
plt.title('Execution Time vs Loss Function')
plt.xlabel('Loss Function')
plt.ylabel('Execution Time (s)')

# Final loss vs loss function
plt.subplot(3, 2, 3)
plt.plot(list(loss_functions.keys()), final_loss_list, marker='o')
plt.title('Final Loss vs Loss Function')
plt.xlabel('Loss Function')
plt.ylabel('Final Loss')

# Final train accuracy vs loss function
plt.subplot(3, 2, 4)
plt.plot(list(loss_functions.keys()), final_train_acc_list, marker='o')
plt.title('Final Train Accuracy vs Loss Function')
plt.xlabel('Loss Function')
plt.ylabel('Final Train Accuracy')

# Final test accuracy vs loss function
plt.subplot(3, 2, 5)
plt.plot(list(loss_functions.keys()), final_test_acc_list, marker='o')
plt.title('Final Test Accuracy vs Loss Function')
plt.xlabel('Loss Function')
plt.ylabel('Final Test Accuracy')

# Parameter statistics vs loss function
plt.subplot(3, 2, 6)
plt.errorbar(list(loss_functions.keys()), param_means, yerr=param_stds, fmt='o')
plt.title('Parameter Statistics vs Loss Function')
plt.xlabel('Loss Function')
plt.ylabel('Parameter Value')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Loss_Function_Comparison.png"), dpi=300)
plt.show()
