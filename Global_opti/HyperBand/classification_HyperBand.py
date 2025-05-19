import optuna
import torch
import pennylane as qml
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from optuna.pruners import HyperbandPruner

# Create directory for results
output_dir = "classification_HyperBand"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Redirect print statements to a .txt file
import sys
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(output_dir, "output_log.txt"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger()

print("Start of the HyperBand Optimization")

# Initialize the quantum device
dev = qml.device("default.qubit", wires=1)

# Load data
train_loader = torch.load("loaders/qml_essai_train_loader.pth")
test_loader = torch.load("loaders/qml_essai_train_loader.pth")

# Configuration settings
batch_size = 20
nb_epoch = 30

def normalize_data(x, normalization):
    if normalization == '-pi_to_pi':
        return (x - x.min()) / (x.max() - x.min()) * 2 * np.pi - np.pi
    elif normalization == '0_to_2pi':
        return (x - x.min()) / (x.max() - x.min()) * 2 * np.pi
    return x

def L1(yh, gt):
    return torch.abs(yh - gt)

def L2(yh, gt):
    return (yh - gt) ** 2

def Huber(yh, gt, delta=1.0):
    abs_error = torch.abs(yh - gt)
    quadratic = torch.minimum(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + delta * linear

def classif(yh):
    if yh < -0.33:
        return -1.0
    if yh > 0.33:
        return 1.0
    return 0.0

@qml.qnode(dev, interface="torch")
def gamma(params, x, depth, input_size):
    for i in range(depth):
        for j in range(input_size):
            qml.RX(params[j * 3  + i * input_size * 3], wires=0)
            qml.RY((params[j * 3  + 1 + i * input_size * 3]) * (x[0][j].item()), wires=0)
            qml.RX(params[j * 3 + 2 + i * input_size * 3], wires=0)
    return qml.expval(qml.PauliZ(0))

def objective(trial):
    # Suggest hyperparameters
    depth = trial.suggest_int('depth', 1, 10)
    learning_rate = trial.suggest_categorical('learning_rate', [0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005, 0.0000005])
    loss_function = trial.suggest_categorical('loss_function', ['L1', 'L2', 'Huber'])
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'Adagrad', 'Adadelta'])
    normalization = trial.suggest_categorical('normalization', ['-pi_to_pi', '0_to_2pi'])

    print(f"Trial {trial.number}: depth={depth}, learning_rate={learning_rate}, loss_function={loss_function}, optimizer={optimizer_name}, normalization={normalization}")

    # Determine the input size
    for x_train, y_train in train_loader:
        input_size = x_train.shape[1]
        break

    # Initialize parameters
    nb_params = depth * 3 * input_size
    params = torch.tensor([0.5] * nb_params, requires_grad=True, dtype=torch.float64)
    
    if optimizer_name == 'Adam':
        opt = torch.optim.Adam([params], lr=learning_rate)
    elif optimizer_name == 'SGD':
        opt = torch.optim.SGD([params], lr=learning_rate)
    elif optimizer_name == 'Adagrad':
        opt = torch.optim.Adagrad([params], lr=learning_rate)
    elif optimizer_name == 'Adadelta':
        opt = torch.optim.Adadelta([params], lr=learning_rate)
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    all_results = []
    
    for epoch in range(nb_epoch):
        for x_train, y_train in train_loader:
            # Normalize data
            x_train_norm = normalize_data(x_train, normalization)
            
            opt.zero_grad()
            res = gamma(params, x_train_norm, depth, input_size)
            scal_res = res

            # Convert ground truth to tensor
            y_train_tensor = torch.tensor(y_train.numpy()[0][0], dtype=torch.float64)

            # Apply the selected loss function
            if loss_function == 'L1':
                loss = L1(scal_res, y_train_tensor)
            elif loss_function == 'Huber':
                loss = Huber(scal_res, y_train_tensor)
            else:  # L2 by default
                loss = L2(scal_res, y_train_tensor)
                
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            
            cl = classif(scal_res.item())
            if cl == y_train_tensor.item():
                epoch_acc += 1

        # Average accuracy and loss
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)
        
        # Calculate test accuracy
        test_acc = 0.0
        test_count = 0
        for x_test, y_test in test_loader:
            x_test_norm = normalize_data(x_test, normalization)
            res_test = gamma(params, x_test_norm, depth, input_size)
            scal_res_test = res_test.item()
            if classif(scal_res_test) == y_test.numpy()[0][0]:
                test_acc += 1
            test_count += 1
        avg_test_acc = test_acc / test_count

        print(f"Epoch {epoch}: Average Loss={avg_loss:.6f}, Train Accuracy={avg_acc:.6f}, Test Accuracy={avg_test_acc:.6f}")
        all_results.append([epoch, avg_loss, avg_acc, avg_test_acc])
    
    # Save raw outputs to CSV
    csv_file_path = os.path.join(output_dir, f"trial_{trial.number}_results.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average Loss', 'Train Accuracy', 'Test Accuracy'])
        writer.writerows(all_results)
    
    # Generate and save plots
    epochs = [result[0] for result in all_results]
    losses = [result[1] for result in all_results]
    train_accuracies = [result[2] for result in all_results]
    test_accuracies = [result[3] for result in all_results]
    
    plt.figure()
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Trial {trial.number}: Loss per Epoch')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"trial_{trial.number}_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Trial {trial.number}: Accuracy per Epoch')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"trial_{trial.number}_accuracy.png"))
    plt.close()

    # Return the final test accuracy (could also return train accuracy or both)
    final_test_acc = test_accuracies[-1]
    return final_test_acc

# Run the HyperBand optimization
study = optuna.create_study(direction="maximize", pruner=HyperbandPruner())
study.optimize(objective, n_trials=100)

print("HyperBand Optimization Complete")
print("Best hyperparameters: ", study.best_params)

# Save the best parameters to a .txt file
with open(os.path.join(output_dir, "best_hyperparameters.txt"), "w") as f:
    f.write(str(study.best_params))
