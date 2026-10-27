"""Exact NumPy simulators and adjoint gradients. Angles are in radians."""
import numpy as np


def rotate(r, axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    a, b = [(1, 2), (2, 0), (0, 1)][axis]
    out = r.copy()
    out[:, a] = c * r[:, a] - s * r[:, b]
    out[:, b] = s * r[:, a] + c * r[:, b]
    return out


class QRU:
    def __init__(self, depth=4, dimensions=3, parameters=3, axes='XYX'):
        self.depth, self.dimensions, self.parameters = depth, dimensions, parameters
        self.axes = tuple('XYZ'.index(a) for a in axes)
        self.size = depth * dimensions * parameters

    def initial(self, seed=0):
        return np.random.default_rng(seed).normal(0.5, 0.1, self.size)

    def gates(self, p, x):
        n = self.parameters
        for k in range(self.depth * self.dimensions):
            j, base = k % self.dimensions, k * n
            yield self.axes[0], p[base], [(base, 1.)]
            if n < 3:
                angle, deriv = x[:, j], []
            elif n == 3:
                angle, deriv = p[base + 1] * x[:, j], [(base + 1, x[:, j])]
            elif n == 4:
                angle = p[base + 1] * x[:, j] + p[base + 2]
                deriv = [(base + 1, x[:, j]), (base + 2, 1.)]
            else:
                angle = p[base + 1] * x[:, j] ** 2 + p[base + 2] * x[:, j] + p[base + 3]
                deriv = [(base + 1, x[:, j] ** 2), (base + 2, x[:, j]), (base + 3, 1.)]
            yield self.axes[1], angle, deriv
            if n >= 2:
                yield self.axes[2], p[base + n - 1], [(base + n - 1, 1.)]

    def forward(self, p, x, cache=False):
        r = np.zeros((len(x), 3)); r[:, 2] = 1.
        trace = []
        for axis, angle, deriv in self.gates(p, x):
            r = rotate(r, axis, angle)
            if cache:
                trace.append((axis, angle, deriv, r))
        return (r[:, 2], trace) if cache else r[:, 2]

    def backward(self, p, x, trace, dy):
        g = np.zeros(self.size)
        adj = np.zeros((len(x), 3)); adj[:, 2] = dy
        for axis, angle, deriv, r in reversed(trace):
            a, b = [(1, 2), (2, 0), (0, 1)][axis]
            da = -adj[:, a] * r[:, b] + adj[:, b] * r[:, a]
            for index, factor in deriv:
                g[index] += np.sum(da * factor)
            adj = rotate(adj, axis, -angle)
        return g


PAULI = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]], complex),
         np.diag([1., -1.]).astype(complex)]


def embedded(a, q):
    mats = [np.eye(2, dtype=complex) for _ in range(3)]; mats[q] = a
    return np.kron(np.kron(mats[0], mats[1]), mats[2])


def cnot(control, target):
    out = np.zeros((8, 8), complex)
    for i in range(8):
        j = i ^ (1 << (2 - target)) if i & (1 << (2 - control)) else i
        out[j, i] = 1.
    return out


class VQC:
    """One RY encoding per input, then RX/RY/RZ and a CNOT ring."""
    def __init__(self, depth=4):
        self.depth, self.size = depth, depth * 9
        self.generators = [embedded(PAULI[a], q) for q in range(3) for a in range(3)]
        self.ring = cnot(2, 0) @ cnot(1, 2) @ cnot(0, 1)
        agg = cnot(2, 0) @ cnot(1, 0)
        self.observable = agg.conj().T @ embedded(PAULI[2], 0) @ agg

    def initial(self, seed=0):
        return np.random.default_rng(seed).normal(0.5, 0.1, self.size)

    @staticmethod
    def states(x):
        s = np.ones((len(x), 1))
        for j in range(3):
            s = (s[:, :, None] * np.stack([np.cos(x[:, j] / 2), np.sin(x[:, j] / 2)], axis=1)[:, None, :]).reshape(len(x), -1)
        return s

    def forward(self, p, x, cache=False):
        u = np.eye(8, dtype=complex); trace = []
        for k, theta in enumerate(p):
            gen = self.generators[k % 9]
            gate = np.cos(theta / 2) * np.eye(8) - 1j * np.sin(theta / 2) * gen
            if cache:
                trace.append((k, gate, gen, u))
            u = gate @ u
            if k % 9 == 8:
                if cache:
                    trace.append((None, self.ring, None, u))
                u = self.ring @ u
        s = self.states(x)
        a = u.conj().T @ self.observable @ u
        y = np.einsum('bi,ij,bj->b', s, a, s, optimize=True).real
        return (y, (trace, s)) if cache else y

    def backward(self, p, x, cache, dy):
        trace, s = cache
        rho = (s.T * dy) @ s
        obs = self.observable.copy(); grad = np.zeros(self.size)
        for k, gate, gen, prefix in reversed(trace):
            if k is not None:
                r = prefix @ rho @ prefix.conj().T
                dgate = -0.5j * gen @ gate
                grad[k] = 2 * np.trace(obs @ dgate @ r @ gate.conj().T).real
            obs = gate.conj().T @ obs @ gate
        return grad


class MLP:
    """Three inputs, one ReLU hidden layer and a scalar tanh head."""
    def __init__(self, hidden=17, dimensions=3):
        self.hidden, self.dimensions = hidden, dimensions
        self.size = (dimensions + 2) * hidden + 1

    def unpack(self, p):
        h, d = self.hidden, self.dimensions
        return p[:d*h].reshape(d, h), p[d*h:(d+1)*h], p[(d+1)*h:(d+2)*h], p[-1]

    def initial(self, seed=0):
        rng = np.random.default_rng(seed); p = np.zeros(self.size)
        p[:self.dimensions*self.hidden] = rng.normal(0, np.sqrt(2/self.dimensions), self.dimensions*self.hidden)
        p[(self.dimensions+1)*self.hidden:-1] = rng.normal(0, 1/np.sqrt(self.hidden), self.hidden)
        return p

    def forward(self, p, x, cache=False):
        w, b, v, c = self.unpack(p); z = x @ w + b
        y = np.tanh(np.maximum(z, 0) @ v + c)
        return (y, (z, y)) if cache else y

    def backward(self, p, x, cache, dy):
        z, y = cache; w, b, v, c = self.unpack(p)
        dt = dy * (1-y*y); dz = dt[:, None] * v[None, :] * (z > 0)
        return np.r_[(x.T @ dz).ravel(), dz.sum(0), np.maximum(z, 0).T @ dt, dt.sum()]


def loss_and_derivative(pred, y, kind='L2'):
    r = pred-y
    if kind == 'L1':
        return np.mean(abs(r)), np.sign(r) / len(r)
    if kind == 'Huber':
        return np.mean(np.where(abs(r) <= 1, .5*r*r, abs(r)-.5)), np.clip(r, -1, 1)/len(r)
    return np.mean(r*r), 2*r/len(r)


def classify(y):
    return np.where(y < -.33, -1, np.where(y > .33, 1, 0))
