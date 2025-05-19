# -*- coding: utf-8 -*-
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
import os
import csv
from scipy.interpolate import UnivariateSpline
import time

# Create output directory
output_dir = "classification_normalisation"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

# Quantum node definition
@qml.qnode(dev, interface="torch")
def gamma(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RX(params[j * 3  + i * input_size * 3], wires=0)
            qml.RY((params[j * 3  + 1 + i * input_size * 3]) * (x[0][j].item()), wires=0)
            qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

def train_model(params, train_loader, test_loader, depth, batch_size, nb_epoch, normalization, output_dir):
    # Training loop
    log_file = open(os.path.join(output_dir, f"train_qml_{normalization}.txt"), "w")
    loss_hist, acc_hist, test_acc_hist = [], [], []
    raw_output = []

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
                raw_output.append([epoch, bcnt, avg_loss, avg_acc, test_acc_fin])
        
                bcnt += 1
        
        epoch += 1
        
    end_time_process = time.time() 
    time_process = end_time_process - start_time_process

    log_file.close()

    # Save raw output to CSV
    with open(os.path.join(output_dir, f"raw_output_{normalization}.csv"), "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['epoch', 'bcnt', 'avg_loss', 'avg_acc', 'test_acc_fin'])
        csvwriter.writerows(raw_output)

    return loss_hist, acc_hist, test_acc_hist, time_process

# Normalization -pi to pi
def normalize_minus_pi_pi(x):
    return -np.pi + (x - x.min()) * (2 * np.pi) / (x.max() - x.min())

# Normalization 0 to 2pi
def normalize_0_2pi(x):
    return (x - x.min()) * (2 * np.pi) / (x.max() - x.min())

# Function to denormalize data
def denormalize_data(x):
    # Apply the formula corresponding to each feature
    x[:, 0] = x[:, 0] * 3  # Feature 1
    x[:, 1] = x[:, 1] * 0.06  # Feature 2
    x[:, 2] = x[:, 2] * 1.75  # Feature 3
    return x

# Normalize data for -pi to pi
for x_train, y_train in train_loader:
    x_train_np = x_train.numpy()
    x_train_np_norm_pi = normalize_minus_pi_pi(x_train_np)
    x_train_np_norm_2pi = normalize_0_2pi(x_train_np)
    x_train_np_denorm = denormalize_data(x_train_np)
    x_train_norm_pi = torch.tensor(x_train_np_norm_pi, dtype=torch.float64)
    x_train_norm_2pi = torch.tensor(x_train_np_norm_2pi, dtype=torch.float64)
    x_train_denorm = torch.tensor(x_train_np_denorm, dtype=torch.float64)
    break

# Train and evaluate for -pi to pi normalization
params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
opt = torch.optim.Adam([params], lr=0.00005)
loss_hist_pi, acc_hist_pi, test_acc_hist_pi, time_process_pi = train_model(params, train_loader, test_loader, depth, batch_size, nb_epoch, "minus_pi_pi", output_dir)

# Train and evaluate for 0 to 2pi normalization
params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
opt = torch.optim.Adam([params], lr=0.00005)
loss_hist_2pi, acc_hist_2pi, test_acc_hist_2pi, time_process_2pi = train_model(params, train_loader, test_loader, depth, batch_size, nb_epoch, "0_2pi", output_dir)

# Train and evaluate for denormalized data (non-normalized)
params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
opt = torch.optim.Adam([params], lr=0.00005)
loss_hist_denorm, acc_hist_denorm, test_acc_hist_denorm, time_process_denorm = train_model(params, train_loader, test_loader, depth, batch_size, nb_epoch, "denormalized", output_dir)

# Plot the distribution of all parameters
plt.hist(params.detach().cpu().numpy(), bins=50, alpha=0.75)
plt.title('Distribution of parameters values')
plt.xlabel('Parameter value')
plt.ylabel('Frequency')
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
loss_hist_ma_pi = moving_average(loss_hist_pi, window_size)
acc_hist_ma_pi = moving_average(acc_hist_pi, window_size)
test_acc_hist_ma_pi = moving_average(test_acc_hist_pi, window_size)

loss_hist_ma_2pi = moving_average(loss_hist_2pi, window_size)
acc_hist_ma_2pi = moving_average(acc_hist_2pi, window_size)
test_acc_hist_ma_2pi = moving_average(test_acc_hist_2pi, window_size)

loss_hist_ma_denorm = moving_average(loss_hist_denorm, window_size)
acc_hist_ma_denorm = moving_average(acc_hist_denorm, window_size)
test_acc_hist_ma_denorm = moving_average(test_acc_hist_denorm, window_size)

batches_per_epoch = 1316

# Convert batch count to epochs
epochs_pi = np.arange(len(loss_hist_ma_pi)) / batches_per_epoch
epochs_2pi = np.arange(len(loss_hist_ma_2pi)) / batches_per_epoch
epochs_denorm = np.arange(len(loss_hist_ma_denorm)) / batches_per_epoch

# Valeurs finales des moyennes mobiles sur les 100 dernières valeurs
final_loss_ma_pi = final_moving_average(loss_hist_ma_pi, final_window_size)
final_train_acc_ma_pi = final_moving_average(acc_hist_ma_pi, final_window_size)
final_test_acc_ma_pi = final_moving_average(test_acc_hist_ma_pi, final_window_size)

final_loss_ma_2pi = final_moving_average(loss_hist_ma_2pi, final_window_size)
final_train_acc_ma_2pi = final_moving_average(acc_hist_ma_2pi, final_window_size)
final_test_acc_ma_2pi = final_moving_average(test_acc_hist_ma_2pi, final_window_size)

final_loss_ma_denorm = final_moving_average(loss_hist_ma_denorm, final_window_size)
final_train_acc_ma_denorm = final_moving_average(acc_hist_ma_denorm, final_window_size)
final_test_acc_ma_denorm = final_moving_average(test_acc_hist_ma_denorm, final_window_size)

# Calculer la trainabilité (aire sous la courbe de la loss)
trainability_pi = np.trapz(loss_hist_ma_pi, epochs_pi)
trainability_2pi = np.trapz(loss_hist_ma_2pi, epochs_2pi)
trainability_denorm = np.trapz(loss_hist_ma_denorm, epochs_denorm)

# Afficher la trainabilité
print(f'Trainability -pi to pi: {trainability_pi:.6f}')
print(f'Trainability 0 to 2pi: {trainability_2pi:.6f}')
print(f'Trainability denormalized: {trainability_denorm:.6f}')

# Tracé des courbes
plt.figure(figsize=(14, 6))

# Graphique des pertes avec courbe ajustée pour -pi à pi
plt.subplot(2, 2, 1)
plt.plot(epochs_pi, loss_hist_ma_pi, label='Loss -pi to pi')
plt.title('Loss -pi to pi')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Graphique des accuracies avec courbes ajustées pour -pi à pi
plt.subplot(2, 2, 2)
plt.plot(epochs_pi, acc_hist_ma_pi, label='Train accuracy -pi to pi')
plt.plot(epochs_pi, test_acc_hist_ma_pi, label='Test accuracy -pi to pi')
plt.title('Accuracy -pi to pi')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Graphique des pertes avec courbe ajustée pour 0 à 2pi
plt.subplot(2, 2, 3)
plt.plot(epochs_2pi, loss_hist_ma_2pi, label='Loss 0 to 2pi')
plt.title('Loss 0 to 2pi')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Graphique des accuracies avec courbes ajustées pour 0 à 2pi
plt.subplot(2, 2, 4)
plt.plot(epochs_2pi, acc_hist_ma_2pi, label='Train accuracy 0 to 2pi')
plt.plot(epochs_2pi, test_acc_hist_ma_2pi, label='Test accuracy 0 to 2pi')
plt.title('Accuracy 0 to 2pi')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Ajouter la courbe pour les données dénormalisées
plt.subplot(2, 2, 1)
plt.plot(epochs_denorm, loss_hist_ma_denorm, label='Loss denormalized')
plt.legend(loc='upper right')

plt.subplot(2, 2, 2)
plt.plot(epochs_denorm, acc_hist_ma_denorm, label='Train accuracy denormalized')
plt.plot(epochs_denorm, test_acc_hist_ma_denorm, label='Test accuracy denormalized')
plt.legend(loc='lower right')

plt.savefig(os.path.join(output_dir, "Loss_Accuracy_Comparison_with_Denormalized.png"), dpi=300)
plt.show()

# Mise à jour du graphique comparatif
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Trainability
axs[0, 0].bar(['-pi to pi', '0 to 2pi', 'Denormalized'], [trainability_pi, trainability_2pi, trainability_denorm], color=['blue', 'orange', 'green'])
axs[0, 0].set_title('Trainability')

# Execution time
axs[0, 1].bar(['-pi to pi', '0 to 2pi', 'Denormalized'], [time_process_pi, time_process_2pi, time_process_denorm], color=['blue', 'orange', 'green'])
axs[0, 1].set_title('Execution Time (seconds)')

# Final loss
axs[0, 2].bar(['-pi to pi', '0 to 2pi', 'Denormalized'], [final_loss_ma_pi, final_loss_ma_2pi, final_loss_ma_denorm], color=['blue', 'orange', 'green'])
axs[0, 2].set_title('Final Mean Loss')

# Final train accuracy
axs[1, 0].bar(['-pi to pi', '0 to 2pi', 'Denormalized'], [final_train_acc_ma_pi, final_train_acc_ma_2pi, final_train_acc_ma_denorm], color=['blue', 'orange', 'green'])
axs[1, 0].set_title('Final Mean Train Accuracy')

# Final test accuracy
axs[1, 1].bar(['-pi to pi', '0 to 2pi', 'Denormalized'], [final_test_acc_ma_pi, final_test_acc_ma_2pi, final_test_acc_ma_denorm], color=['blue', 'orange', 'green'])
axs[1, 1].set_title('Final Mean Test Accuracy')

# Hide the empty subplot
axs[1, 2].axis('off')

plt.savefig(os.path.join(output_dir, "Comparison_Summary_with_Denormalized.png"), dpi=300)
plt.show()
