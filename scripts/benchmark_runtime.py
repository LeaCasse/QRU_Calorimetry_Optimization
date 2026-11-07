"""Serial timings on one host, independent of the parallel experiment jobs."""
import argparse
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import scipy
import sklearn
from qru.data import ROOT,load_data
from qru.train import make_model,fit
from csv_results import read_rows, write_rows


def main(depth_sweep=False):
    d=load_data()
    if depth_sweep:
        selected = {f'QRU{depth}': {'kind': 'QRU', 'depth': depth, 'lr': .003}
                    for depth in [1, 2, 3, 4, 6, 10, 14]}
        output = ROOT/'results/runtime_depth.csv'
    else:
        selected = {}
        for row in read_rows(ROOT/'results/selected.csv'):
            selected[row.pop('model')] = row
        output = ROOT/'results/runtime.csv'
    rows=[]
    for name,c in selected.items():
        m=make_model(**c)
        for seed in range(3):
            p,info=fit(m,d['x_train'],d['y_train'],d['x_val'],d['y_val'],seed=seed,epochs=50,lr=c['lr'])
            for _ in range(3):m.forward(p,d['x_test'])
            timings=[]
            for _ in range(10):
                t=time.perf_counter();m.forward(p,d['x_test']);timings.append(time.perf_counter()-t)
            rows.append({'model':name,'seed':seed,'train_seconds':info['seconds'],
                         'inference_seconds':float(np.median(timings)),'events':len(d['x_test'])})
            print(rows[-1],flush=True)
    metadata = {'python':platform.python_version(), 'numpy':np.__version__,
                'scipy':scipy.__version__, 'sklearn':sklearn.__version__,
                'platform':platform.platform(), 'threads':1, 'epochs':50,
                'timing':'serial, full training and validation each epoch, data loading excluded'}
    write_rows(output, (dict(row, **metadata) for row in rows))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth-sweep', action='store_true')
    main(parser.parse_args().depth_sweep)
