# -*- coding: utf-8 -*-
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
from scipy.interpolate import UnivariateSpline
import time
import torch.nn.functional as F
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

# Create directory for storing results
output_dir = "classification_cross_entropy"
os.makedirs(output_dir, exist_ok=True)

# Initialize log and CSV files
log_file_path = os.path.join(output_dir, "training_log.txt")
csv_file_path = os.path.join(output_dir, "results.csv")

log_file = open(log_file_path, "w")
csv_file = open(csv_file_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "bcnt", "loss", "acc", "test_acc"])

# Loss and classification functions
def gaussian(x, mean, std):
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * torch.sqrt(torch.tensor(2 * np.pi)))

def classif(yh):
    means = torch.tensor([-1.0, 0.0, 1.0])
    std = 0.5
    probs = torch.stack([gaussian(yh, mean, std) for mean in means])
    return probs / probs.sum()

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

# Training loop
loss_hist, acc_hist, test_acc_hist = [], [], []
epoch, bcnt, bloss, bacc, cnt = 0, 0, 0.0, 0, 0

start_time_process = time.time()
while epoch < nb_epoch:
    for x_train, y_train in train_loader:
        opt.zero_grad()
        res = gamma(params, x_train, depth)
        probs = classif(res)
        y_train_prob = torch.tensor([0.0, 0.0, 0.0])
        y_train_prob[int(y_train.item() + 1)] = 1.0  # Assuming y_train is -1, 0, or 1
        loss = F.cross_entropy(probs.unsqueeze(0), y_train_prob.unsqueeze(0))
        loss.backward()
        opt.step()
        bloss += loss.item()
        
        predicted_class = torch.argmax(probs).item() - 1
        if predicted_class == y_train.item():
            bacc += 1
            
        cnt += 1

        # Print progress for each batch
        print(f"Epoch {epoch}, Batch {cnt}: Loss = {loss.item():.6f}, Predicted Class = {predicted_class}, True Class = {y_train.item()}")
        log_file.write(f"Epoch {epoch}, Batch {cnt}: Loss = {loss.item():.6f}, Predicted Class = {predicted_class}, True Class = {y_train.item()}\n")
    
        if cnt == batch_size:
            
            avg_loss = bloss / batch_size
            avg_acc = bacc / batch_size
            loss_hist.append(avg_loss)
            acc_hist.append(avg_acc)
            
            # Evaluate on test set
            test_acc, test_count, test_stop = 0, 0, 100
    
            for x_test, y_test in test_loader:
                res_test = gamma(params, x_test, depth)
                test_probs = classif(res_test)
                if torch.argmax(test_probs).item() - 1 == y_test.item():
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
            csv_writer.writerow([epoch, bcnt, avg_loss, avg_acc, test_acc_fin])
    
            bcnt += 1
    
    epoch += 1

end_time_process = time.time() 
time_process = end_time_process - start_time_process

log_file.close()
csv_file.close()

# Conversion du temps de traitement en jours, heures, minutes et secondes
days = time_process // (24 * 3600)
hours = (time_process % (24 * 3600)) // 3600
minutes = (time_process % 3600) // 60
seconds = time_process % 60

# Affichage du temps de traitement formaté
print(f"Execution time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.4f} seconds")

print(params)

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
loss_hist_ma = moving_average(loss_hist, window_size)
acc_hist_ma = moving_average(acc_hist, window_size)
test_acc_hist_ma = moving_average(test_acc_hist, window_size)

batches_per_epoch = 1316

# Convert batch count to epochs
epochs = np.arange(len(loss_hist_ma)) / batches_per_epoch

# Valeurs finales des moyennes mobiles sur les 100 dernières valeurs
final_loss_ma = final_moving_average(loss_hist_ma, final_window_size)
final_train_acc_ma = final_moving_average(acc_hist_ma, final_window_size)
final_test_acc_ma = final_moving_average(test_acc_hist_ma, final_window_size)

# Calculer la trainabilité (aire sous la courbe de la loss)
trainability = np.trapz(loss_hist_ma, epochs)

# Afficher la trainabilité
print(f'Trainability: {trainability:.6f}')

# Tracé des courbes
plt.figure(figsize=(14, 6))

# Graphique des pertes avec courbe ajustée
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_hist_ma, label='Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.annotate(f'Final mean loss: {final_loss_ma:.6f}', xy=(epochs[-1], loss_hist_ma[-1]), 
             xytext=(epochs[0]+4, final_loss_ma+0.2),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))

# Graphique des accuracies avec courbes ajustées
plt.subplot(1, 2, 2)
plt.plot(epochs, acc_hist_ma, label='Train accuracy')
plt.plot(epochs, test_acc_hist_ma, label='Test accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.annotate(f'Final mean train acc: {final_train_acc_ma:.6f}', xy=(epochs[-1], acc_hist_ma[-1]), 
             xytext=(epochs[0]+4, final_train_acc_ma -0.08),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
plt.annotate(f'Final mean test acc: {final_test_acc_ma:.6f}', xy=(epochs[-1], test_acc_hist_ma[-1]), 
             xytext=(epochs[0]+4, final_test_acc_ma - 0.1),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))

plt.savefig(os.path.join(output_dir, "Loss_Accuracy.png"), dpi=300)
plt.show()
