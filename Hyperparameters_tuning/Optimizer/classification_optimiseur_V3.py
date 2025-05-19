# -*- coding: utf-8 -*-
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
import os
from scipy.interpolate import UnivariateSpline
import time

# Création du répertoire "classification_optimiseurs" s'il n'existe pas
output_dir = "classification_optimiseurs"
os.makedirs(output_dir, exist_ok=True)

# Rediriger les sorties vers un fichier .log dans le répertoire
log_file_path = os.path.join(output_dir, 'classification_optimisateur.log')
log_file = open(log_file_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

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

# Function to run training with a given optimizer
def run_training(optimizer_class, optimizer_params):
    # Initialize parameters as a single tensor
    nb_params = depth * 3 * input_size
    params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
    opt = optimizer_class([params], **optimizer_params)

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
    f = open(os.path.join(output_dir, f"train_qml_{optimizer_class.__name__}.txt"), "w")
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
                f.write(msg + "\n")
        
                bcnt += 1
        
        epoch += 1
        
    end_time_process = time.time() 
    time_process = end_time_process - start_time_process

    f.close()

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
    plt.savefig(os.path.join(output_dir, f"Distribution_of_parameters_values_{optimizer_class.__name__}.png"), dpi=300)
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

    # Graphique des accuracies avec courbes ajustées
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_hist_ma, label='Train accuracy')
    plt.plot(epochs, test_acc_hist_ma, label='Test accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(output_dir, f"Loss_Accuracy_{optimizer_class.__name__}.png"), dpi=300)
    plt.show()

    return {
        'Execution_time': time_process,
        'Final_mean_train_acc': final_train_acc_ma,
        'Final_mean_test_acc': final_test_acc_ma,
        'Final_mean_loss': final_loss_ma,
        'Trainability': trainability,
        'Params': params.detach().cpu().numpy(),
        'Params_mean': params.mean().item(),
        'Params_std': params.std().item()
    }

# List of optimizers to evaluate
optimizers = [
    (torch.optim.SGD, {'lr': 0.01}),
    (torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
    (torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9, 'nesterov': True}),
    (torch.optim.RMSprop, {'lr': 0.01}),
    (torch.optim.Adam, {'lr': 0.001}),
    (torch.optim.Adamax, {'lr': 0.002}),
    (torch.optim.NAdam, {'lr': 0.002}),
    (torch.optim.Adagrad, {'lr': 0.01}),
    (torch.optim.Adadelta, {'lr': 1.0}),
    (torch.optim.AdamW, {'lr': 0.001})
]

results = {}
for optimizer_class, optimizer_params in optimizers:
    print(f"Running training with {optimizer_class.__name__}")
    results[optimizer_class.__name__] = run_training(optimizer_class, optimizer_params)

# Display and compare the results
import pandas as pd

results_df = pd.DataFrame(results).T

# Save results to a CSV file in the output directory
results_df.to_csv(os.path.join(output_dir, "optimizers_comparison.csv"))

# Plot comparison of metrics
plt.figure(figsize=(14, 10))

plt.subplot(3, 2, 1)
plt.bar(results_df.index, results_df['Execution_time'])
plt.title('Execution Time')
plt.xticks(rotation=45)

plt.subplot(3, 2, 2)
plt.bar(results_df.index, results_df['Final_mean_train_acc'])
plt.title('Final Train Accuracy')
plt.xticks(rotation=45)

plt.subplot(3, 2, 3)
plt.bar(results_df.index, results_df['Final_mean_test_acc'])
plt.title('Final Test Accuracy')
plt.xticks(rotation=45)

plt.subplot(3, 2, 4)
plt.bar(results_df.index, results_df['Final_mean_loss'])
plt.title('Final Loss')
plt.xticks(rotation=45)

plt.subplot(3, 2, 5)
plt.bar(results_df.index, results_df['Trainability'])
plt.title('Trainability')
plt.xticks(rotation=45)

plt.subplot(3, 2, 6)
plt.errorbar(results_df.index, results_df['Params_mean'], yerr=results_df['Params_std'], fmt='o')
plt.title('Parameter Mean and Std')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Optimizers_Comparison.png"), dpi=300)
plt.show()

# Fermer le fichier log
log_file.close()
