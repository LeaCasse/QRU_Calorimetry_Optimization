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

# Create directory for results
result_dir = "classification_hybridation"
os.makedirs(result_dir, exist_ok=True)

# Redirect outputs to a .log file in the result directory
log_file_path = os.path.join(result_dir, 'classification_hybridation.log')
log_file = open(log_file_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

# Initialize the quantum device
dev = qml.device("default.qubit", wires=3)

# Load data
prefix = "qml"
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_test_loader.pth")

# Configuration settings
depth = 6
batch_size = 20
nb_epoch = 30

# Determine the input size
for x_train, y_train in train_loader:
    input_size = x_train.shape[1]
    break

nb_params = 54

# Loss and classification functions
def L2(yh, gt):
    return (yh - gt) ** 2

def classif(yh):
    if yh < -0.33:
        return -1.0
    if yh > 0.33:
        return 1.0
    return 0.0

# Define the quantum circuits
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

# Dictionary to store results for each circuit
results = []
df = pd.DataFrame(columns=["circuit", "epoch", "batch", "loss", "train_accuracy", "test_accuracy"])

# Loop over each circuit and train
for circuit_name, gamma in zip(["gamma_1", "gamma_2", "gamma_3"], [gamma_1, gamma_2, gamma_3]):


    params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.Adam([params], lr=0.00005)

    # Training loop
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

                # Add metrics to dataframe
                new_row = {
                    "circuit": circuit_name,
                    "epoch": epoch,
                    "batch": bcnt,
                    "loss": avg_loss,
                    "train_accuracy": avg_acc,
                    "test_accuracy": test_acc_fin
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

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

    # Plot the distribution of all parameters
    plt.hist(params.detach().cpu().numpy(), bins=50, alpha=0.75)
    plt.title(f'Distribution of parameters values for {circuit_name}')
    plt.xlabel('Parameter value')
    plt.ylabel('Frequency')
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

    # Convert batch count to epochs
    epochs = np.arange(len(loss_hist_ma)) / batches_per_epoch

    # Valeurs finales des moyennes mobiles sur les 100 dernières valeurs
    final_loss_ma = final_moving_average(loss_hist_ma, final_window_size)
    final_train_acc_ma = final_moving_average(acc_hist_ma, final_window_size)
    final_test_acc_ma = final_moving_average(test_acc_hist_ma, final_window_size)

    # Calculer la trainabilité (aire sous la courbe de la loss)
    trainability = np.trapz(loss_hist_ma, epochs)

    # Afficher la trainabilité
    print(f'Trainability for {circuit_name}: {trainability:.6f}')

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

    plt.savefig(os.path.join(result_dir, f"Loss_Accuracy_{circuit_name}.png"), dpi=300)
    plt.show()

    # Store results
    results.append({
        "circuit": circuit_name,
        "trainability": trainability,
        "execution_time": execution_time,
        "final_loss_ma": final_loss_ma,
        "final_train_acc_ma": final_train_acc_ma,
        "final_test_acc_ma": final_test_acc_ma
    })

# Print or save the results as needed
for result in results:
    print(f"Results for {result['circuit']}:")
    print(result)

# Save the dataframe to a CSV file
csv_file_path = os.path.join(result_dir, 'classification_hybridation_results.csv')
df.to_csv(csv_file_path, index=False)

# Save summary results to another CSV file
summary_df = pd.DataFrame(results)
summary_csv_file_path = os.path.join(result_dir, 'classification_hybridation_summary.csv')
summary_df.to_csv(summary_csv_file_path, index=False)

# Fermer le fichier log
log_file.close()
