import os
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import pandas as pd

# Initialize the quantum device
dev = qml.device("default.qubit", wires=1)

# Load data
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_test_loader.pth")

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

def denormalize(x):
    ranges = [(0, np.pi), (0, 0.06), (0, 1.75)]
    x_denorm = x.clone()
    for i in range(x.shape[1]):
        x_denorm[:, i] = x[:, i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
    return x_denorm

def normalize_to_pi(x):
    ranges = [(-np.pi, np.pi)] * x.shape[1]
    x_norm = x.clone()
    for i in range(x.shape[1]):
        x_norm[:, i] = (x[:, i] - ranges[i][0]) / (ranges[i][1] - ranges[i][0]) * (2 * np.pi) - np.pi
    return x_norm

def train_and_evaluate(normalized):
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
            if normalized == 'denormalized':
                x_train = denormalize(x_train)
            elif normalized == 'pi_to_neg_pi':
                x_train = normalize_to_pi(x_train)
            
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
                    if normalized == 'denormalized':
                        x_test = denormalize(x_test)
                    elif normalized == 'pi_to_neg_pi':
                        x_test = normalize_to_pi(x_test)
                    
                    res_test = gamma(params, x_test, depth)
                    if classif(res_test.detach().numpy()) == y_test.numpy()[0][0]:
                        test_acc += 1
                    test_count += 1
                    if test_count == test_stop:
                        break
                test_acc_fin = test_acc / test_stop
                test_acc_hist.append(test_acc_fin)
        
                bloss, bacc, cnt = 0.0, 0, 0
        
                bcnt += 1
        
        # Print accuracy and loss at the end of each epoch
        avg_loss = np.mean(loss_hist)
        avg_acc = np.mean(acc_hist)
        print(f"Epoch {epoch+1}/{nb_epoch} - Loss: {avg_loss:.6f}, Train Accuracy: {avg_acc:.6f}, Test Accuracy: {test_acc_fin:.6f}")

        epoch += 1
        
    end_time_process = time.time() 
    time_process = end_time_process - start_time_process

    # Convert batch count to epochs
    batches_per_epoch = 1316
    epochs = np.arange(len(loss_hist)) / batches_per_epoch

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

    # Valeurs finales des moyennes mobiles sur les 100 dernières valeurs
    final_loss_ma = final_moving_average(loss_hist_ma, final_window_size)
    final_train_acc_ma = final_moving_average(acc_hist_ma, final_window_size)
    final_test_acc_ma = final_moving_average(test_acc_hist_ma, final_window_size)

    # Calculer la trainabilité (aire sous la courbe de la loss)
    trainability = np.trapz(loss_hist_ma, epochs)

    return {
        'time': time_process,
        'final_loss': final_loss_ma,
        'train_accuracy': final_train_acc_ma,
        'test_accuracy': final_test_acc_ma,
        'trainability': trainability,
        'loss_hist': loss_hist,
        'acc_hist': acc_hist,
        'test_acc_hist': test_acc_hist
    }

# Create output directory
output_dir = 'classification_normalisation'
os.makedirs(output_dir, exist_ok=True)

# Initialize result storage
results = {}
raw_data = []

def save_results():
    # Save text results
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        for case, metrics in results.items():
            f.write(f"\nCas : {case}\n")
            f.write(f"Temps d'exécution : {metrics['time']:.2f} secondes\n")
            f.write(f"Loss finale : {metrics['final_loss']:.6f}\n")
            f.write(f"Accuracy d'entraînement : {metrics['train_accuracy']:.6f}\n")
            f.write(f"Accuracy de test : {metrics['test_accuracy']:.6f}\n")
            f.write(f"Trainabilité : {metrics['trainability']:.6f}\n")

    # Save raw data to CSV
    df = pd.DataFrame(raw_data, columns=['Case', 'Time', 'Final Loss', 'Train Accuracy', 'Test Accuracy', 'Trainability'])
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)

    # Save plots
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for case in results:
        plt.plot(np.arange(len(results[case]['loss_hist'])) / batches_per_epoch, results[case]['loss_hist'], label=case)
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    for case in results:
        plt.plot(np.arange(len(results[case]['acc_hist'])) / batches_per_epoch, results[case]['acc_hist'], label=f'{case} (Train)')
        plt.plot(np.arange(len(results[case]['test_acc_hist'])) / batches_per_epoch, results[case]['test_acc_hist'], label=f'{case} (Test)')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(output_dir, 'Loss_Accuracy.png'), dpi=300)
    plt.show()

# Execute and collect results
for case in ['0to2pi', 'denormalized', 'pi_to_neg_pi']:
    print(f"\nProcessing case: {case}")
    if case == '0to2pi':
        results[case] = train_and_evaluate(normalized=None)
    elif case == 'denormalized':
        results[case] = train_and_evaluate(normalized='denormalized')
    elif case == 'pi_to_neg_pi':
        results[case] = train_and_evaluate(normalized='pi_to_neg_pi')
    
    # Collect raw data for CSV
    raw_data.append([
        case,
        results[case]['time'],
        results[case]['final_loss'],
        results[case]['train_accuracy'],
        results[case]['test_accuracy'],
        results[case]['trainability']
    ])

# Save all results and plots
save_results()

print("Tous les résultats ont été enregistrés dans le répertoire 'classification_normalisation'.")
