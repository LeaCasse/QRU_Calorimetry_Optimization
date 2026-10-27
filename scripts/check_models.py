"""Independent statevector checks, gradient checks and partition checks."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from qru.models import QRU, VQC, MLP, PAULI, loss_and_derivative
from qru.data import ROOT, load_data


def statevector_qru(model, p, x):
    s = np.zeros((len(x), 2), complex); s[:, 0] = 1
    for axis, angle, _ in model.gates(p, x):
        angle = np.broadcast_to(angle, len(x))
        s = np.cos(angle[:, None]/2)*s-1j*np.sin(angle[:, None]/2)*(s @ PAULI[axis].T)
    return abs(s[:, 0])**2-abs(s[:, 1])**2


def main():
    rng = np.random.default_rng(88); x = rng.normal(size=(19, 3)); y = rng.choice([-1., 0., 1.], 19)
    errors = {}
    models = [QRU(4), QRU(2, parameters=1), QRU(2, parameters=2), QRU(2, parameters=4),
              QRU(2, parameters=5), QRU(2, axes='XYZ'), VQC(3), MLP()]
    for i, model in enumerate(models):
        p = model.initial(9); pred, cache = model.forward(p, x, True)
        _, dy = loss_and_derivative(pred, y); grad = model.backward(p, x, cache, dy)
        fd = np.zeros(model.size)
        for k in range(model.size):
            a, b = p.copy(), p.copy(); a[k] += 1e-6; b[k] -= 1e-6
            fd[k] = (loss_and_derivative(model.forward(a, x), y)[0]-loss_and_derivative(model.forward(b, x), y)[0])/2e-6
        err = float(max(abs(grad-fd))); assert err < 1e-7
        errors[str(i)+'_'+type(model).__name__] = err
        if isinstance(model, QRU):
            err = float(max(abs(pred-statevector_qru(model, p, x)))); assert err < 1e-12
            errors[str(i)+'_statevector'] = err
    model = QRU(5); p = model.initial(1)
    errors['global_parity'] = float(max(abs(model.forward(p, x)-model.forward(p, -x))))
    assert errors['global_parity'] < 1e-12
    # Axis conjugation gives equal families after changing the middle signs.
    p2 = p.reshape(-1, 3).copy(); p2[:, 1] *= -1
    assert max(abs(model.forward(p, x)-QRU(5, axes='YXY').forward(p2.ravel(), x))) < 1e-12
    d = load_data(); keys = [{r.tobytes() for r in d[k]} for k in ['raw_train','raw_val','raw_test']]
    assert all(not keys[i].intersection(keys[j]) for i in range(3) for j in range(i))
    assert np.isfinite(d['x_train']).all() and set(np.unique(d['y_train'])) == {-1, 0, 1}
    record = {'maximum_errors': errors, 'counts': {k: len(d[k]) for k in ['x_train','x_val','x_test']},
              'removed_train_test_feature_overlap': d['removed_overlap'],
              'disjoint_feature_vectors': True, 'train_minimum': d['minimum'].tolist(),
              'train_maximum': d['maximum'].tolist()}
    np.savez_compressed(ROOT/'results/split_indices.npz',train=d['train_indices'],validation=d['val_indices'],
                        minimum=d['minimum'],maximum=d['maximum'],split_seed=np.array(2026))
    (ROOT/'results/validation.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
