# -*- coding: utf-8 -*-
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import time

# Create directory for saving results
directory = "classification_nb_params_per_input"
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize the quantum device
dev = qml.device("default.qubit", wires=1)

# Load data
prefix = "qml"
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_test_loader.pth")

# Configuration settings
depth = 10
batch_size = 20
nb_epoch = 30

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

# Quantum node definitions for different parameter counts
def create_gamma(nb_params_per_input):
    @qml.qnode(dev, interface="torch")
    def gamma(params, x, depth):
        input_size = x.shape[1]  # x should be of shape [batch_size, input_size]
        for i in range(depth):
            for j in range(input_size):
                if nb_params_per_input == 1:
                    qml.RX(params[j + i * input_size], wires=0)
                    qml.RY(x[0][j].item(), wires=0)  # Ensure x[0][j] is a scalar
                elif nb_params_per_input == 2:
                    qml.RX(params[j * 2 + i * input_size * 2], wires=0)
                    qml.RY(x[0][j].item(), wires=0)  # Ensure x[0][j] is a scalar
                    qml.RX(params[j * 2 + 1 + i * input_size * 2], wires=0)
                elif nb_params_per_input == 3:
                    qml.RX(params[j * 3 + i * input_size * 3], wires=0)
                    qml.RY(params[j * 3 + 1 + i * input_size * 3] * x[0][j].item(), wires=0)  # Ensure x[0][j] is a scalar
                    qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
                elif nb_params_per_input == 4:
                    qml.RX(params[j * 4 + i * input_size * 4], wires=0)
                    qml.RY(params[j * 4 + 1 + i * input_size * 4] * x[0][j].item() + params[j * 4 + 2 + i * input_size * 4], wires=0)  # Ensure x[0][j] is a scalar
                    qml.RX(params[j * 4 + 3 + i * input_size * 4], wires=0)
                elif nb_params_per_input == 5:
                    qml.RX(params[j * 5 + i * input_size * 5], wires=0)
                    qml.RY(params[j * 5 + 1 + i * input_size * 5]**2 * x[0][j].item() + params[j * 5 + 2 + i * input_size * 5] * x[0][j].item() + params[j * 5 + 3 + i * input_size * 5], wires=0)  # Ensure x[0][j] is a scalar
                    qml.RX(params[j * 5 + 4 + i * input_size * 5], wires=0)
        return qml.expval(qml.PauliZ(0))
    return gamma

# Function to train and evaluate the model
def train_and_evaluate(nb_params_per_input, log_file):
    nb_params = depth * nb_params_per_input * input_size
    params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.Adam([params], lr=0.00005)
    
    gamma = create_gamma(nb_params_per_input)
    
    # Training loop
    loss_hist, acc_hist, test_acc_hist = [], [], []
    epoch, bcnt, bloss, bacc, cnt = 0, 0, 0.0, 0, 0

    start_time_process = time.time()
    while epoch < nb_epoch:
        for x_train, y_train in train_loader:
            opt.zero_grad()
            res = gamma(params, x_train, depth)
            scal_res = res.detach().numpy()
            cl = classif(scal_res)
            cl_gt = y_train.numpy()[0][0]
            loss = L2(res, y_train)
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
                log_file.write(msg + "\n")

                bcnt += 1

        epoch += 1
    
    end_time_process = time.time() 
    time_process = end_time_process - start_time_process
    
    # Calculer la trainabilité (aire sous la courbe de la loss)
    batches_per_epoch = len(train_loader)
    epochs = np.arange(len(loss_hist)) / batches_per_epoch
    trainability = np.trapz(loss_hist, epochs)
    
    return {
        "loss_hist": loss_hist,
        "acc_hist": acc_hist,
        "test_acc_hist": test_acc_hist,
        "final_loss": np.mean(loss_hist[-10:]),
        "final_train_acc": np.mean(acc_hist[-10:]),
        "final_test_acc": np.mean(test_acc_hist[-10:]),
        "trainability": trainability,
        "execution_time": time_process
    }

# Open log file
log_file_path = os.path.join(directory, "training_log.txt")
log_file = open(log_file_path, "w")

# Test with different number of parameters per input
results = {}
for nb_params_per_input in range(1, 6):
    print(f"Testing with {nb_params_per_input} parameter(s) per input")
    log_file.write(f"Testing with {nb_params_per_input} parameter(s) per input\n")
    results[nb_params_per_input] = train_and_evaluate(nb_params_per_input, log_file)

log_file.close()

# Save raw results to CSV
csv_file_path = os.path.join(directory, "raw_results.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Params per Input", "Final Loss", "Final Train Accuracy", "Final Test Accuracy", "Trainability", "Execution Time"])
    for nb_params_per_input, result in results.items():
        writer.writerow([
            nb_params_per_input,
            result["final_loss"],
            result["final_train_acc"],
            result["final_test_acc"],
            result["trainability"],
            result["execution_time"]
        ])

# Plot the results
fig, axs = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('Comparison of Results for Different Numbers of Parameters per Input')

for i, (nb_params_per_input, result) in enumerate(results.items()):
    row, col = divmod(i, 2)
    
    # Plot loss history
    axs[row, col].plot(result["loss_hist"], label='Loss')
    axs[row, col].set_title(f'{nb_params_per_input} params/input')
    axs[row, col].set_xlabel('Batches')
    axs[row, col].set_ylabel('Loss')
    axs[row, col].legend()
    
    # Plot accuracy history
    axs[row + 1, col].plot(result["acc_hist"], label='Train Accuracy')
    axs[row + 1, col].plot(result["test_acc_hist"], label='Test Accuracy')
    axs[row + 1, col].set_title(f'{nb_params_per_input} params/input')
    axs[row + 1, col].set_xlabel('Batches')
    axs[row + 1, col].set_ylabel('Accuracy')
    axs[row + 1, col].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.95)
fig_path = os.path.join(directory, "Comparison_Results.png")
plt.savefig(fig_path, dpi=300)
plt.show()

# Create a summary table
summary_table = {
    "Params per Input": [],
    "Final Loss": [],
    "Final Train Accuracy": [],
    "Final Test Accuracy": [],
    "Trainability": [],
    "Execution Time": []
}

for nb_params_per_input, result in results.items():
    summary_table["Params per Input"].append(nb_params_per_input)
    summary_table["Final Loss"].append(result["final_loss"])
    summary_table["Final Train Accuracy"].append(result["final_train_acc"])
    summary_table["Final Test Accuracy"].append(result["final_test_acc"])
    summary_table["Trainability"].append(result["trainability"])
    summary_table["Execution Time"].append(result["execution_time"])

summary_df = pd.DataFrame(summary_table)
summary_csv_path = os.path.join(directory, "summary_results.csv")
summary_df.to_csv(summary_csv_path, index=False)

print("Training and evaluation complete. Results saved to the directory:", directory)
