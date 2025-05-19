# -*- coding: utf-8 -*-
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import csv
import time

# Create directory
output_dir = "classification_batch_size_V3"
os.makedirs(output_dir, exist_ok=True)
print(f"Created directory: {output_dir}")

# Initialize the quantum device
dev = qml.device("default.qubit", wires=1)

# Load data
prefix = "qml"
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_train_loader.pth")
print("Data loaded successfully")

# Configuration settings
depth = 10
batch_size = 20
nb_epoch = 30

# Determine the input size
for x_train, y_train in train_loader:
    input_size = x_train.shape[1]
    break

print(f"Input size determined: {input_size}")

# Loss and classification functions
def L2(yh, gt):
    return (yh - gt) ** 2

def classif(yh):
    if yh < -0.33:
        return -1.0
    if yh > 0.33:
        return 1.0
    return 0.0

# Initialize parameters as a single tensor
nb_params = depth * 3 * input_size
params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
print("Parameters initialized")

# Quantum node definition
@qml.qnode(dev, interface="torch")
def gamma(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RX(params[j * 3 + i * input_size * 3], wires=0)
            qml.RY((params[j * 3 + 1 + i * input_size * 3]) * (x[0][j].item()), wires=0)
            qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

# Function to run the training loop
def train_model(opt_freq, opt, params, depth, input_size, train_loader, test_loader, nb_epoch):
    print(f"Training model with opt_freq={opt_freq}")
    loss_hist, acc_hist, test_acc_hist = [], [], []
    epoch, bcnt, bloss, bacc, cnt = 0, 0, 0.0, 0, 0

    start_time_process = time.time()
    while epoch < nb_epoch:
        for x_train, y_train in train_loader:
            res = gamma(params, x_train, depth)
            scal_res = res.detach().numpy()
            cl = classif(scal_res)
            cl_gt = y_train.numpy()[0][0]
            loss = L2(res, y_train)
            loss.backward()
            cnt += 1

            c = loss.detach().numpy()[0][0]
            bloss += c
            if cl == cl_gt:
                bacc += 1

            if cnt % opt_freq == 0:
                # Update weights after every opt_freq steps
                opt.step()
                opt.zero_grad()

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
                bcnt += 1

        epoch += 1
        print(f"Epoch {epoch}/{nb_epoch} completed")

    end_time_process = time.time()
    time_process = end_time_process - start_time_process

    print(f"Training completed for opt_freq={opt_freq} in {time_process:.4f} seconds")
    return loss_hist, acc_hist, test_acc_hist, time_process

# Function to compute moving averages
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to compute the final moving average
def final_moving_average(data, final_window_size):
    return np.mean(data[-final_window_size:])

# Different optimization frequencies to test
opt_freqs = [1, 5, 20, 100, 'thrice', 'twice', 'once']
results = {}

# Train and evaluate for each optimization frequency
for opt_freq in opt_freqs:
    params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.Adam([params], lr=0.00005)

    if opt_freq == 'thrice':
        opt_freq_val = len(train_loader) // 3
    elif opt_freq == 'twice':
        opt_freq_val = len(train_loader) // 2
    elif opt_freq == 'once':
        opt_freq_val = len(train_loader)
    else:
        opt_freq_val = opt_freq

    loss_hist, acc_hist, test_acc_hist, time_process = train_model(opt_freq_val, opt, params, depth, input_size, train_loader, test_loader, nb_epoch)

    # Calculate metrics
    loss_hist_ma = moving_average(loss_hist, 1000)
    acc_hist_ma = moving_average(acc_hist, 1000)
    test_acc_hist_ma = moving_average(test_acc_hist, 1000)
    batches_per_epoch = len(train_loader)
    epochs = np.arange(len(loss_hist_ma)) / batches_per_epoch
    final_loss_ma = final_moving_average(loss_hist_ma, 10)
    final_train_acc_ma = final_moving_average(acc_hist_ma, 10)
    final_test_acc_ma = final_moving_average(test_acc_hist_ma, 10)
    trainability = np.trapz(loss_hist_ma, epochs)

    results[opt_freq] = {
        'loss_hist': loss_hist,
        'acc_hist': acc_hist,
        'test_acc_hist': test_acc_hist,
        'final_loss_ma': final_loss_ma,
        'final_train_acc_ma': final_train_acc_ma,
        'final_test_acc_ma': final_test_acc_ma,
        'trainability': trainability,
        'execution_time': time_process
    }

# Save raw outputs to CSV
csv_file_path = os.path.join(output_dir, "raw_outputs.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Opt_Freq", "Loss_Hist", "Acc_Hist", "Test_Acc_Hist", "Final_Loss_MA", "Final_Train_Acc_MA", "Final_Test_Acc_MA", "Trainability", "Execution_Time"])
    for opt_freq, result in results.items():
        writer.writerow([
            opt_freq,
            result['loss_hist'],
            result['acc_hist'],
            result['test_acc_hist'],
            result['final_loss_ma'],
            result['final_train_acc_ma'],
            result['final_test_acc_ma'],
            result['trainability'],
            result['execution_time']
        ])
print(f"Raw outputs saved to {csv_file_path}")

# Save print outputs to TXT
txt_file_path = os.path.join(output_dir, "train_outputs.txt")
with open(txt_file_path, mode='w') as f:
    for opt_freq, result in results.items():
        f.write(f"Opt Freq: {opt_freq}\n")
        f.write(f"Final Mean Loss: {result['final_loss_ma']:.6f}\n")
        f.write(f"Final Mean Train Accuracy: {result['final_train_acc_ma']:.6f}\n")
        f.write(f"Final Mean Test Accuracy: {result['final_test_acc_ma']:.6f}\n")
        f.write(f"Trainability: {result['trainability']:.6f}\n")
        f.write(f"Execution Time: {result['execution_time']:.4f} seconds\n\n")
print(f"Print outputs saved to {txt_file_path}")

# Plot results for comparison
plt.figure(figsize=(20, 10))
for i, opt_freq in enumerate(opt_freqs):
    result = results[opt_freq]
    epochs = np.arange(len(result['loss_hist'])) / batches_per_epoch

    plt.subplot(2, 4, i + 1)
    plt.plot(epochs, result['loss_hist'], label='Loss')
    plt.plot(epochs, result['acc_hist'], label='Train Acc')
    plt.plot(epochs, result['test_acc_hist'], label='Test Acc')
    plt.title(f'Opt Freq: {opt_freq}')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend(loc='upper right')

plt.tight_layout()
png_file_path = os.path.join(output_dir, "Comparison_Results.png")
plt.savefig(png_file_path, dpi=300)
plt.show()
print(f"Comparison results saved to {png_file_path}")

# Create summary table
summary_table = []
for opt_freq, result in results.items():
    summary_table.append([
        opt_freq,
        result['final_train_acc_ma'],
        result['final_test_acc_ma'],
        result['trainability'],
        result['execution_time'],
        result['final_loss_ma']
    ])

# Save summary table to CSV
summary_csv_file_path = os.path.join(output_dir, "summary_table.csv")
with open(summary_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Opt_Freq", "Final_Train_Acc_MA", "Final_Test_Acc_MA", "Trainability", "Execution_Time", "Final_Loss_MA"])
    writer.writerows(summary_table)
print(f"Summary table saved to {summary_csv_file_path}")
