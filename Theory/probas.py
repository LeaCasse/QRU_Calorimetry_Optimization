import pennylane as qml
import numpy as np
from tqdm import trange

# ------------- configurable -------------
L   = 6      # nombre de layers / re-upload blocks
M   = 100    # tirages aléatoires
N   = 96     # points d'échantillonnage (Nyquist ≥ 2L)
K   = 20     # nombre de fréquences à analyser (0…K-1)
eps = 0.05   # nouveau seuil d'activation
np.random.seed(0)
# ----------------------------------------

# --- devices & circuits -----------------
dev_qru = qml.device("default.qubit", wires=1)
dev_vqc = qml.device("default.qubit", wires=3)

def make_qru():
    @qml.qnode(dev_qru)
    def c(x, th):
        for m in range(L):
            for j in range(3):
                qml.RX(th[m,3*j],     wires=0)
                qml.RY(th[m,3*j+1]*x[j], wires=0)
                qml.RZ(th[m,3*j+2],   wires=0)
        return qml.expval(qml.PauliZ(0))
    return c

def make_vqc():
    @qml.qnode(dev_vqc)
    def c(x, th):
        # data layer
        for w in range(3):
            qml.RY(x[w], wires=w)
        # layers hardware-efficient
        for m in range(L):
            # entanglement
            for w in range(3):
                qml.CNOT(wires=[w, (w+1)%3])
            # rotations
            for w in range(3):
                p0, p1, p2 = th[m,3*w:3*w+3]
                qml.RX(p0, wires=w)
                qml.RY(p1, wires=w)
                qml.RZ(p2, wires=w)
        return qml.expval(qml.PauliZ(0))
    return c

qru = make_qru()
vqc = make_vqc()

# --- échantillonnage le long de la diagonale ---
u  = np.linspace(-1, 1, N)
xu = np.stack([u, u, u], axis=1)

def active_freqs(circuit, K, eps):
    counts = np.zeros(K, dtype=int)
    for _ in trange(M, leave=False):
        th    = np.random.uniform(-np.pi, np.pi, (L, 9))
        f     = np.array([circuit(x, th) for x in xu])
        c_fft = np.fft.fft(f) / N
        for n in range(K):
            if abs(c_fft[n]) > eps:
                counts[n] += 1
    return counts / M

p_qru = active_freqs(qru, K, eps)
p_vqc = active_freqs(vqc, K, eps)

print(f"Activation probability p_n (n=0…{K-1})")
print("n   :", np.arange(K))
print("QRU :", np.round(p_qru, 2))
print("VQC :", np.round(p_vqc, 2))
