"""Training with mean gradients, validation checkpoints and exact evaluation."""
import time
import numpy as np
from .models import QRU, VQC, MLP, classify, loss_and_derivative


def make_model(kind='QRU', depth=4, parameters=3, axes='XYX', dimensions=3, **kwargs):
    if kind == 'VQC':
        return VQC(depth)
    if kind == 'MLP':
        return MLP(kwargs.get('hidden', 17), dimensions)
    return QRU(depth, dimensions, parameters, axes)


class Optimizer:
    def __init__(self, size, kind='Adam', lr=.003):
        self.kind, self.lr, self.t = kind, lr, 0
        self.m = np.zeros(size); self.v = np.zeros(size); self.u = np.zeros(size)

    def step(self, p, g):
        self.t += 1
        if self.kind == 'SGD':
            return p-self.lr*g
        if self.kind == 'Adagrad':
            self.v += g*g
            return p-self.lr*g/(np.sqrt(self.v)+1e-8)
        if self.kind == 'Adadelta':
            self.v = .9*self.v+.1*g*g
            update = np.sqrt(self.u+1e-6)/np.sqrt(self.v+1e-6)*g
            self.u = .9*self.u+.1*update*update
            return p-self.lr*update
        if self.kind == 'RMSprop':
            self.v = .99*self.v+.01*g*g
            return p-self.lr*g/(np.sqrt(self.v)+1e-8)
        self.m = .9*self.m+.1*g; self.v = .999*self.v+.001*g*g
        return p-self.lr*(self.m/(1-.9**self.t))/(np.sqrt(self.v/(1-.999**self.t))+1e-8)


def metrics(pred, y):
    return {'mse': float(np.mean((pred-y)**2)),
            'accuracy': float(np.mean(classify(pred) == y))}


def fit(model, x, y, xv, yv, seed=0, epochs=30, batch_size=256, lr=.003,
        optimizer='Adam', loss='L2', schedule=False, initial=None, evaluate_every=1):
    p = model.initial(seed) if initial is None else initial.copy()
    rng = np.random.default_rng(seed+10000)
    opt = Optimizer(len(p), optimizer, lr)
    best = (np.inf, p.copy(), 0)
    trace = []
    start = time.perf_counter()
    for epoch in range(epochs+1):
        if epoch % evaluate_every == 0 or epoch == epochs:
            tr = metrics(model.forward(p, x), y); va = metrics(model.forward(p, xv), yv)
            trace.append({'epoch': epoch, 'train_mse': tr['mse'], 'train_accuracy': tr['accuracy'],
                          'val_mse': va['mse'], 'val_accuracy': va['accuracy']})
            if va['mse'] < best[0]:
                best = va['mse'], p.copy(), epoch
        if epoch == epochs:
            break
        order = rng.permutation(len(x))
        opt.lr = lr*(.1 if schedule and epoch >= epochs//2 else 1.)*(.1 if schedule and epoch >= 3*epochs//4 else 1.)
        for i in range(0, len(x), batch_size):
            idx = order[i:i+batch_size]
            pred, cache = model.forward(p, x[idx], True)
            _, dy = loss_and_derivative(pred, y[idx], loss)
            grad = model.backward(p, x[idx], cache, dy)
            if not np.isfinite(grad).all():
                raise FloatingPointError('Nonfinite gradient')
            p = opt.step(p, grad)
    elapsed = time.perf_counter()-start
    return best[1], {'trace': trace, 'best_epoch': best[2], 'seconds': elapsed,
                     'steps': epochs*((len(x)+batch_size-1)//batch_size),
                     'parameter_count': model.size, 'seed': seed}
