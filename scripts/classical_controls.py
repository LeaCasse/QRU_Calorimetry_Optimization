"""Classical controls selected on validation, with standard multiclass heads."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import json
import sys
import time
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.exceptions import ConvergenceWarning
from qru.data import ROOT, load_data
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def basis(x):
    phi = np.ones((len(x), 1))
    for j in range(3):
        phi = (phi[:, :, None]*np.stack([np.ones(len(x)),np.sin(x[:,j]),np.cos(x[:,j])], axis=1)[:,None,:]).reshape(len(x),-1)
    return phi


def main():
    d = load_data(); rows, predictions = [], {}
    methods = {
        'Logistic': ([.1, 1., 10.], lambda c, s: LogisticRegression(C=c, max_iter=1000, random_state=s)),
        'Fourier': ([.1, 1., 10.], lambda c, s: LogisticRegression(C=c, max_iter=1000, random_state=s)),
        'Trees': ([3, 7, 15], lambda c, s: HistGradientBoostingClassifier(max_leaf_nodes=c, max_iter=100,
                    learning_rate=.1, early_stopping=False, l2_regularization=1., random_state=s)),
        'MLP_softmax': ([.0005, .003, .03], lambda c, s: MLPClassifier(hidden_layer_sizes=(12,), activation='relu',
                    solver='adam', batch_size=256, learning_rate_init=c, max_iter=50, alpha=0,
                    random_state=s, early_stopping=False, tol=0, n_iter_no_change=1000))}
    for name,(grid, maker) in methods.items():
        if name == 'Fourier':
            x,xv,xt = [basis(d[k]) for k in ['x_train','x_val','x_test']]
        else:
            x,xv,xt = [d['scale'](d[k],'standard') for k in ['raw_train','raw_val','raw_test']]
        choices=[]
        for value in grid:
            m=maker(value,123);m.fit(x,d['y_train']);choices.append(log_loss(d['y_val'],m.predict_proba(xv)))
        selected=grid[int(np.argmin(choices))]
        for seed in (range(10) if name=='MLP_softmax' else [0]):
            start=time.perf_counter();m=maker(selected,seed);m.fit(x,d['y_train']);elapsed=time.perf_counter()-start
            pv,pt=m.predict_proba(xv),m.predict_proba(xt);pred=m.predict(xt)
            rows.append({'method':name,'seed':seed,'selected_value':selected,'validation_log_loss':float(log_loss(d['y_val'],pv)),
                         'test_accuracy':float(accuracy_score(d['y_test'],pred)),'seconds':elapsed,
                         'test_mse':float(np.mean((pt @ m.classes_-d['y_test'])**2)),
                         'parameter_count':87 if name=='MLP_softmax' else (84 if name=='Fourier' else (12 if name=='Logistic' else None))})
            predictions[f'{name}_{seed}_prediction']=pred;predictions[f'{name}_{seed}_probability']=pt
            print(name,seed,rows[-1]['test_accuracy'],flush=True)
        (ROOT/'results/classical.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(ROOT/'results/classical_predictions.npz',**predictions)


if __name__ == '__main__':main()
