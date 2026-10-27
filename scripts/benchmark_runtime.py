"""Serial timings on one host, independent of the parallel experiment jobs."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import json
import platform
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import scipy
import sklearn
from qru.data import ROOT,load_data
from qru.train import make_model,fit


def main():
    d=load_data();selected=json.loads((ROOT/'results/selected.json').read_text());rows=[]
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
            (ROOT/'results/runtime.json').write_text(json.dumps({'runs':rows,'python':platform.python_version(),
                'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
                'platform':platform.platform(),'threads':1,'timing':'serial, full training and validation each epoch, data loading excluded'},indent=2))


if __name__=='__main__':main()
