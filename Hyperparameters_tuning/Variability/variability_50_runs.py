import pennylane as qml
import numpy as np
import torch
import os
import csv
import time

# Configuration du dispositif et des paramètres
dev = qml.device("default.qubit", wires=1)
depth = 5  # Profondeur modifiée
batch_size = 20
lr = 0.5  # Learning rate modifié
nb_epoch = 30
num_runs = 50

# Chargement et configuration des loaders
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_test_loader.pth")

# Fonction de perte : moyenne de l'erreur quadratique
def L2(yh, gt):
    return torch.mean((yh - gt) ** 2)  # Retourne un scalaire

def classif(yh):
    if yh < -0.33:
        return -1.0
    if yh > 0.33:
        return 1.0
    return 0.0

# Définition de la fonction gamma (le modèle quantique)
@qml.qnode(dev, interface="torch")
def gamma(params, x, depth, input_size):
    for i in range(depth):
        for j in range(input_size):
            qml.RX(params[j * 3  + i * input_size * 3], wires=0)
            qml.RY((params[j * 3  + 1 + i * input_size * 3]) * (x[0][j].item()), wires=0)
            qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

# Détermination de la taille d'entrée
for x_train, y_train in train_loader:
    input_size = x_train.shape[1]
    break

# Création du répertoire pour stocker les résultats
output_dir = "variability_50_runs"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "results.csv")

# Initialisation du fichier CSV avec l'en-tête
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Run", "Final_Loss", "Final_Accuracy"])

# Fonction pour un run d'entraînement unique
def run_training():
    params = torch.randn(depth * 3 * input_size, requires_grad=True, dtype=torch.float64)  # Initialisation aléatoire
    opt = torch.optim.Adadelta([params], lr=lr)  # Optimiseur Adadelta
    loss_hist, acc_hist = [], []

    # Boucle d'apprentissage
    for epoch in range(nb_epoch):
        bloss, bacc, cnt = 0.0, 0, 0
        for x_train, y_train in train_loader:
            opt.zero_grad()
            res = gamma(params, x_train, depth, input_size)
            scal_res = res.detach().numpy()
            cl = classif(scal_res)
            cl_gt = y_train.numpy()[0][0]
            loss = L2(res, y_train)  # Calcul de la loss moyenne pour obtenir un scalaire
            loss.backward()
            opt.step()
            bloss += loss.item()  # Utiliser .item() pour extraire la valeur scalaire

            if cl == cl_gt:
                bacc += 1

            cnt += 1
            if cnt == batch_size:
                loss_hist.append(bloss / batch_size)
                acc_hist.append(bacc / batch_size)
                bloss, bacc, cnt = 0.0, 0, 0

    # Calcul des moyennes finales
    final_loss = np.mean(loss_hist[-10:])  # Moyenne des 10 dernières valeurs de loss
    final_acc = np.mean(acc_hist[-10:])    # Moyenne des 10 dernières valeurs d'accuracy
    return final_loss, final_acc

# Boucle sur plusieurs runs pour calculer la variabilité et enregistrer les résultats
for run in range(1, num_runs + 1):
    # Mélange des données
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=True)
    
    final_loss, final_acc = run_training()
    
    # Écriture des résultats dans le fichier CSV et flush pour sécuriser l'écriture
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([run, final_loss, final_acc])
        file.flush()  # Assure que les données sont bien enregistrées immédiatement

print("Les résultats de tous les runs ont été enregistrés dans:", output_file)
