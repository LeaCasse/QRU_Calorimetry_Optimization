# -*- coding: utf-8 -*-
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
from scipy.interpolate import UnivariateSpline
import time
import os
import csv

# Create directory for storing results
output_dir = "classification_GPU"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
opt = torch.optim.Adam([params], lr=0.00005)

# Function to run training
def run_training(device_name, device_label):
    print(f"Starting training on {device_label}")
    dev = qml.device(device_name, wires=1)

    @qml.qnode(dev, interface="torch")
    def gamma(params, x, depth):
        for i in range(depth):
            for j in range(input_size):
                qml.RX(params[j * 3  + i * input_size * 3], wires=0)
                qml.RY((params[j * 3  + 1 + i * input_size * 3]) * (x[0][j].item()), wires=0)
                qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
        return qml.expval(qml.PauliZ(0))

    # Initialize logs
    txt_filename = os.path.join(output_dir, f"train_{device_label}.txt")
    csv_filename = os.path.join(output_dir, f"results_{device_label}.csv")
    with open(txt_filename, "w") as f_txt, open(csv_filename, "w", newline='') as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["epoch", "batch", "loss", "accuracy", "test_accuracy"])

        loss_hist, acc_hist, test_acc_hist = [], [], []
        epoch, bcnt, bloss, bacc, cnt = 0, 0, 0.0, 0, 0

        start_time_process = time.time()
        while epoch < nb_epoch:
            print(f"Epoch {epoch+1}/{nb_epoch}")
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
                    f_txt.write(msg + "\n")
                    csv_writer.writerow([epoch, bcnt, avg_loss, avg_acc, test_acc_fin])

                    bcnt += 1

            epoch += 1

        end_time_process = time.time()
        time_process = end_time_process - start_time_process

    return loss_hist, acc_hist, test_acc_hist, time_process, params

# Run training with and without GPU
loss_hist_cpu, acc_hist_cpu, test_acc_hist_cpu, time_cpu, params_cpu = run_training("default.qubit", "CPU")
loss_hist_gpu, acc_hist_gpu, test_acc_hist_gpu, time_gpu, params_gpu = run_training("lightning.qubit", "GPU")

# Conversion du temps de traitement en jours, heures, minutes et secondes
def format_time(time_process):
    days = time_process // (24 * 3600)
    hours = (time_process % (24 * 3600)) // 3600
    minutes = (time_process % 3600) // 60
    seconds = time_process % 60
    return f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.4f} seconds"

print("CPU Execution time:", format_time(time_cpu))
print("GPU Execution time:", format_time(time_gpu))

# Save execution times to a file
with open(os.path.join(output_dir, "execution_times.txt"), "w") as f:
    f.write(f"CPU Execution time: {format_time(time_cpu)}\n")
    f.write(f"GPU Execution time: {format_time(time_gpu)}\n")

# Plot the distribution of all parameters
plt.hist(params_cpu.detach().cpu().numpy(), bins=50, alpha=0.75, label="CPU")
plt.hist(params_gpu.detach().cpu().numpy(), bins=50, alpha=0.75, label="GPU")
plt.title('Distribution of parameters values')
plt.xlabel('Parameter value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(os.path.join(output_dir, "Distribution_of_parameters_values.png"), dpi=300)
plt.show()

# Fonction pour calculer les moyennes mobiles
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Fonction pour calculer la moyenne des 100 dernières valeurs
def final_moving_average(data, final_window_size):
    return np.mean(data[-final_window_size:])

# Paramètres des fenêtres
window_size = 1000
final_window_size = 10

# Calculer les moyennes mobiles
loss_hist_ma_cpu = moving_average(loss_hist_cpu, window_size)
acc_hist_ma_cpu = moving_average(acc_hist_cpu, window_size)
test_acc_hist_ma_cpu = moving_average(test_acc_hist_cpu, window_size)

loss_hist_ma_gpu = moving_average(loss_hist_gpu, window_size)
acc_hist_ma_gpu = moving_average(acc_hist_gpu, window_size)
test_acc_hist_ma_gpu = moving_average(test_acc_hist_gpu, window_size)

batches_per_epoch = 1316

# Convert batch count to epochs
epochs_cpu = np.arange(len(loss_hist_ma_cpu)) / batches_per_epoch
epochs_gpu = np.arange(len(loss_hist_ma_gpu)) / batches_per_epoch

# Valeurs finales des moyennes mobiles sur les 100 dernières valeurs
final_loss_ma_cpu = final_moving_average(loss_hist_ma_cpu, final_window_size)
final_train_acc_ma_cpu = final_moving_average(acc_hist_ma_cpu, final_window_size)
final_test_acc_ma_cpu = final_moving_average(test_acc_hist_ma_cpu, final_window_size)

final_loss_ma_gpu = final_moving_average(loss_hist_ma_gpu, final_window_size)
final_train_acc_ma_gpu = final_moving_average(acc_hist_ma_gpu, final_window_size)
final_test_acc_ma_gpu = final_moving_average(test_acc_hist_ma_gpu, final_window_size)

# Calculer la trainabilité (aire sous la courbe de la loss)
trainability_cpu = np.trapz(loss_hist_ma_cpu, epochs_cpu)
trainability_gpu = np.trapz(loss_hist_ma_gpu, epochs_gpu)

# Afficher la trainabilité
print(f'CPU Trainability: {trainability_cpu:.6f}')
print(f'GPU Trainability: {trainability_gpu:.6f}')

# Save trainability to a file
with open(os.path.join(output_dir, "trainability.txt"), "w") as f:
    f.write(f'CPU Trainability: {trainability_cpu:.6f}\n')
    f.write(f'GPU Trainability: {trainability_gpu:.6f}\n')

# Tracé des courbes
plt.figure(figsize=(14, 6))

# Graphique des pertes avec courbe ajustée
plt.subplot(1, 2, 1)
plt.plot(epochs_cpu, loss_hist_ma_cpu, label='Loss CPU')
plt.plot(epochs_gpu, loss_hist_ma_gpu, label='Loss GPU')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.annotate(f'Final mean loss CPU: {final_loss_ma_cpu:.6f}', xy=(epochs_cpu[-1], loss_hist_ma_cpu[-1]), 
             xytext=(epochs_cpu[0]+4, final_loss_ma_cpu+0.2),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
plt.annotate(f'Final mean loss GPU: {final_loss_ma_gpu:.6f}', xy=(epochs_gpu[-1], loss_hist_ma_gpu[-1]), 
             xytext=(epochs_gpu[0]+4, final_loss_ma_gpu+0.2),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))

# Graphique des accuracies avec courbes ajustées
plt.subplot(1, 2, 2)
plt.plot(epochs_cpu, acc_hist_ma_cpu, label='Train accuracy CPU')
plt.plot(epochs_gpu, acc_hist_ma_gpu, label='Train accuracy GPU')
plt.plot(epochs_cpu, test_acc_hist_ma_cpu, label='Test accuracy CPU')
plt.plot(epochs_gpu, test_acc_hist_ma_gpu, label='Test accuracy GPU')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.annotate(f'Final mean train acc CPU: {final_train_acc_ma_cpu:.6f}', xy=(epochs_cpu[-1], acc_hist_ma_cpu[-1]), 
             xytext=(epochs_cpu[0]+4, final_train_acc_ma_cpu -0.08),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
plt.annotate(f'Final mean train acc GPU: {final_train_acc_ma_gpu:.6f}', xy=(epochs_gpu[-1], acc_hist_ma_gpu[-1]), 
             xytext=(epochs_gpu[0]+4, final_train_acc_ma_gpu -0.08),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
plt.annotate(f'Final mean test acc CPU: {final_test_acc_ma_cpu:.6f}', xy=(epochs_cpu[-1], test_acc_hist_ma_cpu[-1]), 
             xytext=(epochs_cpu[0]+4, final_test_acc_ma_cpu - 0.1),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
plt.annotate(f'Final mean test acc GPU: {final_test_acc_ma_gpu:.6f}', xy=(epochs_gpu[-1], test_acc_hist_ma_gpu[-1]), 
             xytext=(epochs_gpu[0]+4, final_test_acc_ma_gpu - 0.1),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))

plt.savefig(os.path.join(output_dir, "Loss_Accuracy.png"), dpi=300)
plt.show()
