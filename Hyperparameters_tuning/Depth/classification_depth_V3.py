# -*- coding: utf-8 -*-
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

# Initialize the quantum device
dev = qml.device("default.qubit", wires=1)

# Load data
prefix = "qml"
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_train_loader.pth")

# Configuration settings
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
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

# Function to calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to calculate final moving average
def final_moving_average(data, final_window_size):
    return np.mean(data[-final_window_size:])

# Create directory for outputs
output_dir = "classification_depth_V3"
os.makedirs(output_dir, exist_ok=True)

# File paths for log and CSV files
log_file = os.path.join(output_dir, "train_qml_classification_depth.log")
output_csv = os.path.join(output_dir, "raw_output.csv")

# Initialize logging
with open(log_file, "w") as f:
    results = []
    for depth in depths:
        print(f"Training for depth: {depth}")
        f.write(f"Training for depth: {depth}\n")

        # Initialize parameters as a single tensor
        nb_params = depth * 3 * input_size
        params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
        opt = torch.optim.Adam([params], lr=0.00005)

        # Quantum node definition
        @qml.qnode(dev, interface="torch")
        def gamma(params, x, depth):
            for i in range(depth):
                for j in range(input_size):
                    qml.RX(params[j * 3 + i * input_size * 3], wires=0)
                    qml.RY((params[j * 3 + 1 + i * input_size * 3]) * (x[0][j].item()), wires=0)
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

        # Conversion du temps de traitement en jours, heures, minutes et secondes
        days = time_process // (24 * 3600)
        hours = (time_process % (24 * 3600)) // 3600
        minutes = (time_process % 3600) // 60
        seconds = time_process % 60

        # Affichage du temps de traitement formaté
        print(f"Execution time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.4f} seconds")
        f.write(f"Execution time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.4f} seconds\n")

        # Plot the distribution of all parameters
        plt.figure(figsize=(8, 6))  # Adjust the figure size for better clarity
        plt.hist(params.detach().cpu().numpy(), bins=50, alpha=0.75)
        plt.title('Distribution of parameters values')
        plt.xlabel('Parameter value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f"Distribution_of_parameters_values_depth_{depth}.png"), dpi=300)
        plt.show()

        # Paramètres des fenêtres
        window_size = 1000
        final_window_size = 10

        # Calculer les moyennes mobiles
        loss_hist_ma = moving_average(loss_hist, window_size)
        acc_hist_ma = moving_average(acc_hist, window_size)
        test_acc_hist_ma = moving_average(test_acc_hist, window_size)

        batches_per_epoch = 1316  # Modify according to your actual batch size

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
        f.write(f'Trainability: {trainability:.6f}\n')

        results.append({
            'depth': depth,
            'avg_loss': avg_loss,
            'final_test_acc': final_test_acc_ma,
            'final_train_acc': final_train_acc_ma,
            'execution_time': time_process,
            'trainability': trainability
        })

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
                     xytext=(epochs[0] + 4, final_loss_ma + 0.2),
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
                     xytext=(epochs[0] + 4, final_train_acc_ma - 0.08),
                     arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))
        plt.annotate(f'Final mean test acc: {final_test_acc_ma:.6f}', xy=(epochs[-1], test_acc_hist_ma[-1]), 
                     xytext=(epochs[0] + 4, final_test_acc_ma - 0.1),
                     arrowprops=dict(facecolor='black', width=0.5, headwidth=5, shrink=0.05))

        plt.savefig(os.path.join(output_dir, f"Loss_Accuracy_depth_{depth}.png"), dpi=300)
        plt.show()

    # Save raw output to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)

# Analyze results
depths = [result['depth'] for result in results]
trainabilities = [result['trainability'] for result in results]
times = [result['execution_time'] for result in results]
final_losses = [result['avg_loss'] for result in results]
final_train_accuracies = [result['final_train_acc'] for result in results]
final_test_accuracies = [result['final_test_acc'] for result in results]

# Plot trainability vs depth
plt.figure()
plt.plot(depths, trainabilities, marker='o')
plt.title('Trainability vs Depth')
plt.xlabel('Depth')
plt.ylabel('Trainability')
plt.savefig(os.path.join(output_dir, "Trainability_vs_Depth.png"), dpi=300)
plt.show()

# Plot execution time vs depth
plt.figure()
plt.plot(depths, times, marker='o')
plt.title('Execution Time vs Depth')
plt.xlabel('Depth')
plt.ylabel('Execution Time (s)')
plt.savefig(os.path.join(output_dir, "Execution_Time_vs_Depth.png"), dpi=300)
plt.show()

# Plot final loss vs depth
plt.figure()
plt.plot(depths, final_losses, marker='o')
plt.title('Final Loss vs Depth')
plt.xlabel('Depth')
plt.ylabel('Final Loss')
plt.savefig(os.path.join(output_dir, "Final_Loss_vs_Depth.png"), dpi=300)
plt.show()

# Plot final train accuracy vs depth
plt.figure()
plt.plot(depths, final_train_accuracies, marker='o')
plt.title('Final Train Accuracy vs Depth')
plt.xlabel('Depth')
plt.ylabel('Final Train Accuracy')
plt.savefig(os.path.join(output_dir, "Final_Train_Accuracy_vs_Depth.png"), dpi=300)
plt.show()

# Plot final test accuracy vs depth
plt.figure()
plt.plot(depths, final_test_accuracies, marker='o')
plt.title('Final Test Accuracy vs Depth')
plt.xlabel('Depth')
plt.ylabel('Final Test Accuracy')
plt.savefig(os.path.join(output_dir, "Final_Test_Accuracy_vs_Depth.png"), dpi=300)
plt.show()
