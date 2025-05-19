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

for idx, gamma in enumerate(circuits):
    # Initialiser les métriques
    loss_hist, acc_hist, test_acc_hist = [], [], []
    epoch, bcnt, bloss, bacc, cnt = 0, 0, 0.0, 0, 0

    start_time_process = time.time()
    while epoch < nb_epoch:
        for x_train, y_train in train_loader:
            opt.zero_grad()
            res = gamma(params, x_train[0], depth)  # Note: x_train[0] pour obtenir les valeurs de x
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

    # Évaluation sur le test set
    test_acc = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            res = gamma(params, x_test[0], depth)  # Note: x_test[0] pour obtenir les valeurs de x
            scal_res = res.detach().numpy()
            cl = classif(scal_res)
            cl_gt = y_test.numpy()[0][0]
            test_acc += (1.0 if cl == cl_gt else 0.0)
    test_acc /= len(test_loader)

    # Stockage des résultats
    results.append({
        'circuit': f'gamma_{idx+1}',
        'final_loss': loss_hist[-1],
        'final_train_acc': acc_hist[-1],
        'final_test_acc': test_acc,
        'trainability': np.trapz(loss_hist, dx=1),
        'execution_time': end_time_process - start_time_process,
        'trained_params': params.detach().numpy()
    })

    # Tracer les courbes de loss et d'accuracy
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(len(loss_hist)), loss_hist, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(len(acc_hist)), acc_hist, color='tab:blue', label='Train Accuracy')
    ax2.plot(range(len(acc_hist)), [test_acc] * len(acc_hist), color='tab:green', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f'Loss and Accuracy for Circuit {idx+1}')
    plt.legend()
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
