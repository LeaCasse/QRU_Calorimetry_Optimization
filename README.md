# Optimizing Hyperparameters for Quantum Data Re-Uploaders in Calorimetric Particle Identification

This repository accompanies the research article:

**Léa Cassé**, **Bernhard Pfahringer**, **Albert Bifet**, and **Frédéric Magniette**  
_"Optimizing Hyperparameters for Quantum Data Re-Uploaders in Calorimetric Particle Identification"_

---

## 🧠 Overview

This project explores the expressivity and performance of single-qubit **Quantum Re-uploading Units (QRUs)** for classifying particles in a simulated high-granularity calorimeter dataset.

We study the influence of architectural and training hyperparameters on model accuracy, expressivity (via Fourier analysis), and computational cost. The study also includes global hyperparameter optimization using **Bayesian Optimization** and **HyperBand**.

---

## 📦 Project Structure

```

QRU\_Calorimetry\_Optimization/
├── data/                    # Dataset files (preprocessed calorimeter data)
├── Hyperparameters_tuning/  # For each hyperparameters: models, training, and analysis
├── Global_opti/             # Global optimisation: Bayesian and HyperBand
├── Theory/                  # Theory experiments
├── Comparison_QRU_VQC_MLP   # Coparison between those three models with same trainable parameters budget
├── Optimizing Hyper...      # Paper PDF and related documentation
├── requirements.txt         # Dependencies
└── README.md                # This file

````

---

## 📊 Dataset

We use a subset of the **D2 calorimetry simulation dataset** (Becheva et al., 2024), which models the response of a CMS-like detector to single particles. For this study, we selected:

- **Three classes**: electrons, pions, and muons  
- **Three features**: total ECAL energy, shower length, HCAL energy std deviation

---

## 🧪 Install requirements

```bash
pip install -r requirements.txt
````

---

## ⚙️ Key Features

* **Single-qubit QRU** with re-uploaded classical data via angle rotations
* **Comparative Fourier expressivity** with VQC baselines
* **Systematic tuning** of:

  * Circuit depth
  * Number of parameters per input
  * Rotation gates
  * Normalization ranges
  * Batch size, optimizer, loss function, learning rate
* **Hyperparameter optimization** using:

  * Bayesian optimization (scikit-optimize)
  * HyperBand + fANOVA priors

---

## ✅ Main Results

| Metric                | Value        |
| --------------------- | ------------ |
| Best test accuracy    | 0.985        |
| Optimal circuit       | Rx – Ry – Rx |
| Optimal depth         | 4 to 5       |
| Optimal learning rate | 5e-4         |
| Best optimizer        | Adam         |
| Best loss function    | L2           |

---

## 📚 References

* Pérez-Salinas et al., *Data re-uploading for a universal quantum classifier*, 2019
* Barthe & Pérez-Salinas, *Gradients and frequency profiles of quantum re-uploading models*, 2023
* Becheva et al., *High granularity calorimetry D2 dataset*, 2024
* Cerezo et al., *Variational quantum algorithms*, 2021

---

## 📄 License

This code is released under the MIT License.

---

For any questions or collaboration, feel free to contact:
📧 casse.lea@gmail.com
