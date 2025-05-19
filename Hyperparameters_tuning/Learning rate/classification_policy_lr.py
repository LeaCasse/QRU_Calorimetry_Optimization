# -*- coding: utf-8 -*-
import os
import pennylane as qml
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Créer un répertoire pour stocker les résultats
output_dir = "regression_loss_function"
os.makedirs(output_dir, exist_ok=True)

# Initialisation du dispositif quantique
dev = qml.device("default.qubit", wires=1)

# Chargement des données
print("Chargement des données...")
data = pd.read_csv('Dataregression/electrons.csv', header=None)

# Normalisation des trois premières caractéristiques entre -pi et pi
print("Normalisation des trois premières caractéristiques entre -pi et pi...")
for col in data.columns[:3]:
    data[col] = 2 * np.pi * (data[col] - data[col].min()) / (data[col].max() - data[col].min()) - np.pi

# Extraction des caractéristiques et des énergies
features = data.iloc[:, :-1].values
energies = data.iloc[:, -1].values

# Conversion des données en tensors PyTorch
print("Conversion des données en tensors PyTorch...")
features = torch.tensor(features, dtype=torch.float64)
energies = torch.tensor(energies, dtype=torch.float64).view(-1, 1)

# Division des données en ensembles d'entraînement et de test
print("Division des données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(features, energies, test_size=0.2, random_state=42)

# Paramètres de configuration
nb_epoch = 30  # Nombre d'itérations d'entraînement
input_size = X_train.shape[1]  # Taille d'entrée déterminée à partir des données d'entraînement
lr = 0.001
depth = 10  # Profondeur du circuit quantique
batch_size = 20  # Nombre d'exemples dans chaque lot de données
nb_params = depth * 3 * input_size

# Définition du dataset et du dataloader
print("Création de DataLoader...")
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Définition du circuit quantique
@qml.qnode(dev, interface="torch")
def gamma_1(params, x, depth):
    for i in range(depth):
        for j in range(input_size):
            qml.RX(params[j * 3 + 0 + i * input_size * 3], wires=0)
            qml.RY(params[j * 3 + 1 + i * input_size * 3] * x[j], wires=0)
            qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

# Fonction pour calculer la probabilité gaussienne
def gaussian(x, mean, std):
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * torch.sqrt(2 * torch.tensor(np.pi)))

# Fonction de classification
def classif(yh):
    means = torch.tensor([-1.0, 0.0, 1.0])  # Les classes cibles
    std = 0.5
    probs = torch.stack([gaussian(yh, mean, std) for mean in means])  # Calcul des probabilités gaussiennes
    return probs / probs.sum()  # Normalisation pour que la somme des probabilités soit égale à 1

# Fonction de perte (erreur quadratique moyenne)
def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

# Fonction de perte (erreur absolue)
def l1_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))

# Fonction de perte (Huber)
def huber_loss(y_pred, y_true, delta=1.0):
    l1 = torch.abs(y_pred - y_true)
    l2 = 0.5 * (y_pred - y_true) ** 2
    return torch.mean(torch.where(l1 <= delta, l2, delta * l1 - 0.5 * delta ** 2))

# Fonction de précision pour la régression basée sur la tolérance
def acc(y_pred, y_true, tol=0.1):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean(np.abs((y_pred - y_true) / y_true) <= tol)

# Liste des fonctions de perte à évaluer
loss_functions = {
    'CrossEntropy': F.cross_entropy,
    'L1': l1_loss,
    'L2': mse_loss,
    'Huber': huber_loss
}

# Initialisation des résultats
results_list = []

for loss_name, loss_fn in loss_functions.items():
    print(f"Entraînement avec la fonction de perte : {loss_name}")
    
    # Initialisation des paramètres pour chaque fonction de perte
    params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.Adam([params], lr=lr)
    
    # Boucle d'entraînement
    loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    
    start_time_process = time.time()
    for epoch in range(nb_epoch):
        epoch_loss = 0.0
        train_preds = []
        train_targets = []
        for x_train_batch, y_train_batch in train_loader:
            opt.zero_grad()
            batch_loss = 0.0
            for i in range(len(x_train_batch)):
                x_train = x_train_batch[i]
                y_train = y_train_batch[i]
                x_train = x_train.squeeze()  # Assurer que x_train est de la bonne dimension
                res = gamma_1(params, x_train, depth)
                if loss_name == 'CrossEntropy':
                    probs = classif(res)  # Prédictions du modèle
                    y_train_prob = torch.zeros(3)
                    y_train_prob[int(y_train.item() + 1)] = 1.0  # Étiquettes réelles en one-hot encoding
                    loss = loss_fn(probs.unsqueeze(0), y_train_prob.unsqueeze(0))  # Calcul de la perte
                else:
                    loss = loss_fn(res, y_train)
                loss.backward()
                opt.step()
                batch_loss += loss.item()
                train_preds.append(res.item())
                train_targets.append(y_train.item())
            epoch_loss += batch_loss / len(x_train_batch)
    
        avg_loss = epoch_loss / len(train_loader)
        avg_train_acc = acc(train_preds, train_targets)
        loss_hist.append(avg_loss)
        train_acc_hist.append(avg_train_acc)
    
        # Évaluation sur l'ensemble de test
        test_preds = []
        test_targets = []
        for x_test_batch, y_test_batch in test_loader:
            for i in range(len(x_test_batch)):
                x_test = x_test_batch[i]
                y_test = y_test_batch[i]
                x_test = x_test.squeeze()  # Assurer que x_test est de la bonne dimension
                test_res = gamma_1(params, x_test, depth)
                test_preds.append(test_res.item())
                test_targets.append(y_test.item())
        avg_test_acc = acc(test_preds, test_targets)
        test_acc_hist.append(avg_test_acc)
    
        print(f"Epoch {epoch+1} terminée, Perte Moyenne: {avg_loss:.6f}, Précision d'Entraînement: {avg_train_acc:.6f}, Précision de Test: {avg_test_acc:.6f}")
    
    end_time_process = time.time()
    execution_time = end_time_process - start_time_process
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    dispersion = np.abs(test_targets - test_preds)
    
    # Enregistrement des résultats pour cette fonction de perte
    results = {
        "loss_function": loss_name,
        "execution_time": execution_time,
        "final_train_accuracy": avg_train_acc,
        "final_test_accuracy": avg_test_acc,
        "final_loss": avg_loss,
        "trainability": np.trapz(loss_hist)
    }
    results_list.append(results)
    
    # Sauvegarde des résultats dans un fichier texte
    with open(os.path.join(output_dir, f"results_{loss_name}.txt"), "w") as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")
    
    # Sauvegarde des raw outputs dans un fichier CSV
    raw_output_df = pd.DataFrame({
        'y_true': test_targets,
        'y_pred': test_preds,
        'dispersion': dispersion
    })
    raw_output_df.to_csv(os.path.join(output_dir, f"raw_outputs_{loss_name}.csv"), index=False)
    
    # Traçage des graphiques
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    ax[0].plot(loss_hist, label='Perte d\'Entraînement')
    ax[0].set_xlabel('Époque')
    ax[0].set_ylabel('Perte')
    ax[0].legend()
    ax[0].set_title('Perte d\'Entraînement')
    
    ax[1].plot(train_acc_hist, label='Précision d\'Entraînement')
    ax[1].plot(test_acc_hist, label='Précision de Test')
    ax[1].set_xlabel('Époque')
    ax[1].set_ylabel('Précision')
    ax[1].legend()
    ax[1].set_title('Précision')
    
    plt.suptitle(f'Régression d\'énergie des électrons (gamma_1) - {loss_name}')
    plt.savefig(os.path.join(output_dir, f"Loss_Accuracy_{loss_name}.png"), dpi=300)
    plt.show()
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    ax[0].scatter(test_targets, test_preds, alpha=0.6)
    ax[0].plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], color='red')
    ax[0].set_xlabel('Valeurs Vraies')
    ax[0].set_ylabel('Valeurs Prédites')
    ax[0].set_title('Valeurs Vraies vs Prédites')
    
    ax[1].scatter(test_targets, dispersion, alpha=0.6)
    ax[1].set_xlabel('Valeurs Vraies')
    ax[1].set_ylabel('Dispersion')
    ax[1].set_title('Dispersion Moyenne des Valeurs Prédites')
    
    plt.suptitle(f'Régression d\'énergie des électrons (gamma_1) - {loss_name}')
    plt.savefig(os.path.join(output_dir, f"Energy_regression_electrons_true_pred_dispersion_{loss_name}.png"), dpi=300)
    plt.show()

    # Sauvegarde du graphique de distribution des paramètres
    plt.figure(figsize=(10, 6))
    plt.hist(params.detach().cpu().numpy(), bins=30, alpha=0.75)
    plt.xlabel('Valeurs des Paramètres')
    plt.ylabel('Fréquence')
    plt.title(f'Distribution des Paramètres Entraînés - {loss_name}')
    plt.savefig(os.path.join(output_dir, f"distribution_params_{loss_name}.png"), dpi=300)
    plt.show()

# Tableau comparatif des fonctions de perte
comparison_df = pd.DataFrame(results_list)
comparison_df.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)

# Sauvegarde des prints dans un fichier texte
import sys
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(os.path.join(output_dir, "output_log.txt"))

# Impression des résultats
print(comparison_df)

