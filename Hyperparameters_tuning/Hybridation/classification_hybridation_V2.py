# -*- coding: utf-8 -*-
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
import os
from scipy.interpolate import UnivariateSpline
import time

# Créer un répertoire pour les résultats
result_dir = "classification_hybridation"
os.makedirs(result_dir, exist_ok=True)

# Rediriger les sorties vers un fichier .log dans le répertoire de résultats
log_file_path = os.path.join(result_dir, 'classification_hybridation.log')
log_file = open(log_file_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

# Initialiser le dispositif quantique
dev = qml.device("default.qubit", wires=3)

# Charger les données
prefix = "qml"
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_test_loader.pth")

# Paramètres de configuration
depth = 6
batch_size = 20
nb_epoch = 30

# Déterminer la taille de l'entrée
for x_train, y_train in train_loader:
    input_size = x_train.shape[1]
    break

# Calculer le nombre de paramètres pour chaque circuit
nb_params_gamma_1 = input_size * 3 * depth
nb_params_gamma_2 = input_size * 6 * (depth // 2)
nb_params_gamma_3 = input_size * 9 * (depth // 3)

# Fonctions de perte et de classification
def L2(yh, gt):
    return (yh - gt) ** 2

def classif(yh):
    if yh < -0.33:
        return -1.0
    if yh > 0.33:
        return 1.0
    return 0.0

# Définir les circuits quantiques
@qml.qnode(dev, interface="torch")
def gamma_1(params, x, depth): 
    for i in range(depth):
        for j in range(input_size):
            qml.RX(params[j * 3 + 0 + i * input_size * 3], wires=0)
            qml.RY(params[j * 3 + 1 + i * input_size * 3] * x[0][j].item(), wires=0) 
            qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="torch")
def gamma_2(params, x, depth): 
    for i in range(depth//2):
        for j in range(input_size):
            qml.RX(params[j * 6 + 0 + i * input_size * 6], wires=0)
            qml.RY(params[j * 6 + 1 + i * input_size * 6] * x[0][j].item(), wires=0) 
            qml.RX(params[j * 6 + 2 + i * input_size * 6], wires=0)
            
            qml.RX(params[j * 6 + 3 + i * input_size * 6], wires=1)
            qml.RY(params[j * 6 + 4 + i * input_size * 6] * x[0][j].item(), wires=1) 
            qml.RX(params[j * 6 + 5 + i * input_size * 6], wires=1)

    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="torch")
def gamma_3(params, x, depth): 
    for i in range(depth//3):
        for j in range(input_size):
            qml.RX(params[j * 9 + 0 + i * input_size * 9], wires=0)
            qml.RY(params[j * 9 + 1 + i * input_size * 9] * x[0][j].item(), wires=0) 
            qml.RX(params[j * 9 + 2 + i * input_size * 9], wires=0)
   
            qml.RX(params[j * 9 + 3 + i * input_size * 9], wires=1)
            qml.RY(params[j * 9 + 4 + i * input_size * 9] * x[0][j].item(), wires=1) 
            qml.RX(params[j * 9 + 5 + i * input_size * 9], wires=1)

            qml.RX(params[j * 9 + 6 + i * input_size * 9], wires=2)
            qml.RY(params[j * 9 + 7 + i * input_size * 9] * x[0][j].item(), wires=2) 
            qml.RX(params[j * 9 + 8 + i * input_size * 9], wires=2)

    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[0, 1])
    
    return qml.expval(qml.PauliZ(0))

# Liste pour stocker les résultats de chaque circuit
results = []
rows = []

# Boucle sur chaque circuit et entraînement
for circuit_name, gamma, nb_params in zip(["gamma_1", "gamma_2", "gamma_3"], [gamma_1, gamma_2, gamma_3], [nb_params_gamma_1, nb_params_gamma_2, nb_params_gamma_3]):

    params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.Adam([params], lr=0.00005)

    # Boucle d'entraînement
    train_file_path = os.path.join(result_dir, f"train_{circuit_name}.txt")
    f = open(train_file_path, "w")
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

                # Évaluer sur le jeu de test
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

                # Ajouter les métriques à la liste des lignes
                rows.append({
                    "circuit": circuit_name,
                    "epoch": epoch,
                    "batch": bcnt,
                    "loss": avg_loss,
                    "train_accuracy": avg_acc,
                    "test_accuracy": test_acc_fin
                })

                bcnt += 1

        epoch += 1

    end_time_process = time.time()
    time_process = end_time_process - start_time_process

    # Conversion du temps de traitement en jours, heures, minutes et secondes
    days = time_process // (24 * 3600)
    hours = (time_process % (24 * 3600)) // 3600
    minutes = (time_process % 3600) // 60
    seconds = time_process % 60

    # Affichage du temps de traitement formaté
    execution_time = f"Execution time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.4f} seconds"
    print(execution_time)

    # Tracer la distribution de tous les paramètres
    plt.hist(params.detach().cpu().numpy(), bins=50, alpha=0.75)
    plt.title(f'Distribution des valeurs des paramètres pour {circuit_name}')
    plt.xlabel('Valeur du paramètre')
    plt.ylabel('Fréquence')
    plt.savefig(os.path.join(result_dir, f"Distribution_of_parameters_values_for_{circuit_name}.png"), dpi=300)
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

    # Convertir le nombre de lots en époques
    epochs = np.arange(len(loss_hist_ma)) / batches_per_epoch

    # Valeurs finales des moyennes mobiles sur les 100 dernières valeurs
    final_loss_ma = final_moving_average(loss_hist_ma, final_window_size)
    final_train_acc_ma = final_moving_average(acc_hist_ma, final_window_size)
    final_test_acc_ma = final_moving_average(test_acc_hist_ma, final_window_size)

    # Calculer la trainabilité (aire sous la courbe de la perte)
    trainability = np.trapz(loss_hist_ma, epochs)

    # Afficher la trainabilité
    print(f'Trainability pour {circuit_name}: {trainability:.6f}')

    # Tracer les courbes
    plt.figure(figsize=(14, 6))

    # Graphique des pertes avec courbe ajustée
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_hist_ma, label='Loss')
    plt.title('Perte')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend(loc='upper right')
    plt.annotate(f'Perte moyenne finale: {final_loss_ma:.6f}', xy=(epochs[-1], loss_hist_ma[-1]), 
                 xytext=(epochs[0]+4, final_loss_ma+0.2),
                 arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))

    # Graphique des précisions avec courbes ajustées
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_hist_ma, label='Précision d\'entraînement')
    plt.plot(epochs, test_acc_hist_ma, label='Précision de test')
    plt.title('Précision')
    plt.xlabel('Époques')
    plt.ylabel('Précision')
    plt.legend(loc='lower right')
    plt.annotate(f'Précision moyenne d\'entraînement finale: {final_train_acc_ma:.6f}', xy=(epochs[-1], acc_hist_ma[-1]), 
                 xytext=(epochs[0]+4, final_train_acc_ma -0.08),
                 arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
    plt.annotate(f'Précision moyenne de test finale: {final_test_acc_ma:.6f}', xy=(epochs[-1], test_acc_hist_ma[-1]), 
                 xytext=(epochs[0]+4, final_test_acc_ma - 0.1),
                 arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))

    plt.savefig(os.path.join(result_dir, f"Loss_Accuracy_{circuit_name}.png"), dpi=300)
    plt.show()

    # Stocker les résultats
    results.append({
        "circuit": circuit_name,
        "trainability": trainability,
        "execution_time": execution_time,
        "final_loss_ma": final_loss_ma,
        "final_train_acc_ma": final_train_acc_ma,
        "final_test_acc_ma": final_test_acc_ma
    })

# Après la boucle, convertir la liste des lignes en DataFrame
df = pd.DataFrame(rows)

# Enregistrer le DataFrame dans un fichier CSV
csv_file_path = os.path.join(result_dir, 'classification_hybridation_results.csv')
df.to_csv(csv_file_path, index=False)

# Enregistrer les résultats résumés dans un autre fichier CSV
summary_df = pd.DataFrame(results)
summary_csv_file_path = os.path.join(result_dir, 'classification_hybridation_summary.csv')
summary_df.to_csv(summary_csv_file_path, index=False)

# Tracer la comparaison des métriques
circuit_names = [result['circuit'] for result in results]
trainability_values = [result['trainability'] for result in results]
execution_time_values = [result['execution_time'] for result in results]
final_loss_ma_values = [result['final_loss_ma'] for result in results]
final_train_acc_ma_values = [result['final_train_acc_ma'] for result in results]
final_test_acc_ma_values = [result['final_test_acc_ma'] for result in results]

# Convertir le temps d'exécution de string à secondes pour la comparaison
execution_time_seconds = []
for time_str in execution_time_values:
    days, hours, minutes, seconds = map(float, time_str.replace(' days', '').replace(' hours', '').replace(' minutes', '').replace(' seconds', '').split(','))
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    execution_time_seconds.append(total_seconds)

# Créer des sous-graphiques pour comparer chaque métrique
fig, axs = plt.subplots(3, 2, figsize=(14, 12))

# Trainability
axs[0, 0].bar(circuit_names, trainability_values, color='skyblue')
axs[0, 0].set_title('Trainability')
axs[0, 0].set_ylabel('Trainability (Aire sous la courbe de perte)')

# Temps d'exécution
axs[0, 1].bar(circuit_names, execution_time_seconds, color='lightgreen')
axs[0, 1].set_title('Temps d\'exécution')
axs[0, 1].set_ylabel('Temps d\'exécution (secondes)')

# Perte finale
axs[1, 0].bar(circuit_names, final_loss_ma_values, color='salmon')
axs[1, 0].set_title('Perte moyenne finale')
axs[1, 0].set_ylabel('Perte finale (Moyenne des dernières valeurs)')

# Précision d'entraînement finale
axs[1, 1].bar(circuit_names, final_train_acc_ma_values, color='lightcoral')
axs[1, 1].set_title('Précision finale d\'entraînement')
axs[1, 1].set_ylabel('Précision d\'entraînement (Moyenne des dernières valeurs)')

# Précision de test finale
axs[2, 0].bar(circuit_names, final_test_acc_ma_values, color='gold')
axs[2, 0].set_title('Précision finale de test')
axs[2, 0].set_ylabel('Précision de test (Moyenne des dernières valeurs)')

# Ajuster la disposition et enregistrer la figure
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'Comparison_of_Metrics.png'), dpi=300)
plt.show()

# Fermer le fichier log
log_file.close()
