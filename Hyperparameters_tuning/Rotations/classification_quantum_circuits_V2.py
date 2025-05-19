# -*- coding: utf-8 -*-
import os
import pandas as pd
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# Créer le répertoire pour les résultats
output_dir = "classification_circuits_rotation"
os.makedirs(output_dir, exist_ok=True)

# Rediriger les prints vers un fichier .txt
log_file_path = os.path.join(output_dir, "training_log.txt")
log_file = open(log_file_path, "w")

def print_and_log(message):
    print(message)
    log_file.write(message + "\n")

# Initialisation du dispositif quantique
dev = qml.device("default.qubit", wires=1)

# Chargement des données
prefix = "qml"
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_train_loader.pth")

# Configuration
depth = 10
batch_size = 20
nb_epoch = 30

# Détermination de la taille d'entrée
for x_train, y_train in train_loader:
    input_size = x_train.shape[1]
    break

# Fonctions de perte et de classification
def L2(yh, gt):
    return (yh - gt) ** 2

def classif(yh):
    if yh < -0.33:
        return -1.0
    if yh > 0.33:
        return 1.0
    return 0.0

# Initialisation des paramètres
nb_params = depth * 3 * input_size
params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
opt = torch.optim.Adam([params], lr=0.00005)

# Définition des circuits quantiques avec interface Torch
@qml.qnode(dev, interface='torch')
def gamma_1(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RX(params[j * 3  + i * input_size * 3], wires=0)
            qml.RY((params[j * 3  + 1 + i * input_size * 3]) * x[j], wires=0)
            qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface='torch')
def gamma_2(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RX(params[j * 3  + i * input_size * 3], wires=0)
            qml.RZ((params[j * 3  + 1 + i * input_size * 3]) * x[j], wires=0)
            qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface='torch')
def gamma_3(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RY(params[j * 3  + i * input_size * 3], wires=0)
            qml.RX((params[j * 3  + 1 + i * input_size * 3]) * x[j], wires=0)
            qml.RY(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface='torch')
def gamma_4(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RY(params[j * 3  + i * input_size * 3], wires=0)
            qml.RZ((params[j * 3  + 1 + i * input_size * 3]) * x[j], wires=0)
            qml.RY(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface='torch')
def gamma_5(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RZ(params[j * 3  + i * input_size * 3], wires=0)
            qml.RY((params[j * 3  + 1 + i * input_size * 3]) * x[j], wires=0)
            qml.RZ(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface='torch')
def gamma_6(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RZ(params[j * 3  + i * input_size * 3], wires=0)
            qml.RX((params[j * 3  + 1 + i * input_size * 3]) * x[j], wires=0)
            qml.RZ(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

# Liste des circuits
circuits = [gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6]
results = []

# Fonction pour calculer les moyennes mobiles
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Fonction pour calculer la moyenne des dernières valeurs
def final_moving_average(data, final_window_size):
    return np.mean(data[-final_window_size:])

window_size = 1000
final_window_size = 10
batches_per_epoch = 1316

for idx, gamma in enumerate(circuits):
    # Initialiser les métriques
    loss_hist, acc_hist, test_acc_hist = [], [], []
    epoch, bcnt, bloss, bacc, cnt = 0, 0, 0.0, 0, 0

    start_time_process = time.time()
    while epoch < nb_epoch:
        for x_train, y_train in train_loader:
            opt.zero_grad()
            res = gamma(params, x_train[0], depth)
            scal_res = res.detach().numpy()
            cl = classif(scal_res)
            cl_gt = y_train.numpy()[0][0]
            loss = L2(res, y_train)
            loss.backward()
            opt.step()
            c = loss.detach().numpy()[0]
            bloss += c
            bacc += (1.0 if cl == cl_gt else 0.0)
            cnt += 1
            bcnt += 1
            if bcnt == batch_size:
                print_and_log(f"Epoch {epoch} | Batch {bcnt} | Loss: {bloss/bcnt} | Acc: {bacc/bcnt}")
                loss_hist.append(bloss / bcnt)
                acc_hist.append(bacc / bcnt)
                bloss, bacc, bcnt = 0.0, 0, 0
        epoch += 1

    end_time_process = time.time()

    # Moyennes mobiles
    loss_hist_ma = moving_average(loss_hist, window_size)
    acc_hist_ma = moving_average(acc_hist, window_size)
    test_acc_hist_ma = moving_average(test_acc_hist, window_size)

    epochs = np.arange(len(loss_hist_ma)) / batches_per_epoch

    # Moyennes finales
    final_loss_ma = final_moving_average(loss_hist_ma, final_window_size)
    final_train_acc_ma = final_moving_average(acc_hist_ma, final_window_size)
    final_test_acc_ma = final_moving_average(test_acc_hist_ma, final_window_size)

    # Calcul de la trainabilité
    trainability = np.trapz(loss_hist_ma, epochs)

    # Stockage des résultats
    results.append({
        'circuit': f'gamma_{idx+1}',
        'final_loss': final_loss_ma,
        'final_train_acc': final_train_acc_ma,
        'final_test_acc': final_test_acc_ma,
        'trainability': trainability,
        'execution_time': end_time_process - start_time_process,
        'trained_params': params.detach().numpy()
    })

    # Tracer les courbes de loss et d'accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graphique des pertes
    ax1.plot(epochs, loss_hist_ma, label='Loss')
    ax1.set_title('Perte')
    ax1.set_xlabel('Époques')
    ax1.set_ylabel('Perte')
    ax1.legend(loc='upper right')
    ax1.annotate(f'Perte moyenne finale: {final_loss_ma:.6f}', xy=(epochs[-1], loss_hist_ma[-1]), 
                 xytext=(epochs[0]+4, final_loss_ma+0.2),
                 arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))

    # Graphique des précisions
    ax2.plot(epochs, acc_hist_ma, label='Précision d\'entraînement')
    ax2.plot(epochs, test_acc_hist_ma, label='Précision de test')
    ax2.set_title('Précision')
    ax2.set_xlabel('Époques')
    ax2.set_ylabel('Précision')
    ax2.legend(loc='lower right')
    ax2.annotate(f'Précision moyenne d\'entraînement finale: {final_train_acc_ma:.6f}', xy=(epochs[-1], acc_hist_ma[-1]), 
                 xytext=(epochs[0]+4, final_train_acc_ma -0.08),
                 arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
    ax2.annotate(f'Précision moyenne de test finale: {final_test_acc_ma:.6f}', xy=(epochs[-1], test_acc_hist_ma[-1]), 
                 xytext=(epochs[0]+4, final_test_acc_ma - 0.1),
                 arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))

    fig.tight_layout()
    fig_path = os.path.join(output_dir, f"Loss_Accuracy_Circuit_{idx+1}.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    # Réinitialiser les paramètres pour le prochain circuit
    params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.Adam([params], lr=0.00005)

# Fermeture du fichier log
log_file.close()

# Sauvegarde des résultats bruts dans un fichier .csv
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "raw_results.csv"), index=False)

# Affichage des résultats sous forme de tableau dans Spyder ou Jupyter
print(results_df)

# Visualisation des métriques globales
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparaison des métriques pour différents circuits quantiques')

metrics = ['final_loss', 'final_train_acc', 'final_test_acc', 'trainability', 'execution_time']
metric_labels = ['Loss finale', 'Précision de train finale', 'Précision de test finale', 'Trainability', 'Temps d\'exécution']

for i, metric in enumerate(metrics):
    ax = axs[i//3, i%3]
    values = [result[metric] for result in results]
    ax.bar(range(1, 7), values)
    ax.set_title(metric_labels[i])
    ax.set_xlabel('Circuit')
    ax.set_ylabel(metric_labels[i])
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([result['circuit'] for result in results])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_path = os.path.join(output_dir, "Comparative_Metrics.png")
plt.savefig(fig_path, dpi=300)
plt.show()
