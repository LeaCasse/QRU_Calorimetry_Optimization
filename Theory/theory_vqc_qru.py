import pennylane as qml, numpy as np
from tqdm import trange

# ------------- configurable -------------
L         = 6            # re-upload blocks / layers
M         = 100          # random parameter draws
N         = 96         # sampling points (Nyquist ≥ L)
eps       = 1e-3         # activation threshold
np.random.seed(0)
# ----------------------------------------

# --- devices & circuits -----------------
dev_qru = qml.device("default.qubit", wires=1)
dev_vqc = qml.device("default.qubit", wires=3)

# QRU (RX–RY(x)–RZ)
def make_qru():
    @qml.qnode(dev_qru)
    def c(x, th):
        for m in range(L):
            qml.RX(th[m,0], wires=0); qml.RY(th[m,1]*x[0], wires=0); qml.RZ(th[m,2], wires=0)
            qml.RX(th[m,3], wires=0); qml.RY(th[m,4]*x[1], wires=0); qml.RZ(th[m,5], wires=0)
            qml.RX(th[m,6], wires=0); qml.RY(th[m,7]*x[2], wires=0); qml.RZ(th[m,8], wires=0)
        return qml.expval(qml.PauliZ(0))
    return c

# VQC (one data layer + hardware-efficient)
def make_vqc():
    @qml.qnode(dev_vqc)
    def c(x, th):
        for w in range(3):
            qml.RY(x[w], wires=w)
        for m in range(L):
            for w in range(3):
                qml.CNOT(wires=[w,(w+1)%3])
            for w in range(3):
                p0,p1,p2 = th[m,3*w:3*w+3]
                qml.RX(p0,wires=w); qml.RY(p1,wires=w); qml.RZ(p2,wires=w)
        return qml.expval(qml.PauliZ(0))
    return c

qru = make_qru()
vqc = make_vqc()

# --- sample diagonal --------------------
u = np.linspace(-1,1,N)
xu = np.stack([u,u,u],axis=1)

def active_freqs(circuit):
    """Return boolean activation array for n = 0…L."""
    counts = np.zeros(L+1, dtype=int)
    for _ in trange(M, leave=False):
        th = np.random.uniform(-np.pi, np.pi, (L,9))
        f  = np.array([circuit(x, th) for x in xu])
        c  = np.fft.fft(f) / N                       # unshifted
        for n in range(L+1):
            if abs(c[n]) > eps: counts[n] += 1
    return counts / M

p_qru = active_freqs(qru)
p_vqc = active_freqs(vqc)

print("Activation probability p_n (n=0…L)")
print("n :", np.arange(20))
print("QRU:", np.round(p_qru,2))
print("VQC:", np.round(p_vqc,2))
