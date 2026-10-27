"""Check released predictions against the exact saved models and tables."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import gzip
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from collections import Counter
import numpy as np
from qru.data import ROOT,load_data
from qru.train import make_model,metrics


def main():
    d=load_data()
    with gzip.open(ROOT/'results/runs.json.gz','rt') as f:rows=json.load(f)
    ids=[r['id'] for r in rows];assert len(ids)==len(set(ids))
    counts=Counter(r['config']['stage'] for r in rows)
    assert counts=={'sweeps':138,'comparison':40,'feature_ablation':21,'tuning':12,'bayesian':50,'hyperband':67}
    weights=np.load(ROOT/'results/models.npz',allow_pickle=False)
    max_error=0.;evaluated=0
    for r in rows:
        c=r['config'];features=c.get('features',[0,1,2]);m=make_model(dimensions=len(features),**c)
        p=weights[r['id']+'_theta'];assert len(p)==m.size and np.isfinite(p).all()
        for t in r['fit']['trace']:
            assert all(np.isfinite(t[k]) for k in ['train_mse','val_mse','train_accuracy','val_accuracy'])
        if 'test' not in r:continue
        x=d['scale'](d['raw_test'],c.get('normalization','centered'))[:,features]
        p2=m.forward(p,x);released=weights[r['id']+'_prediction'];error=float(max(abs(p2-released)))
        max_error=max(error,max_error);assert error<1e-12
        measured=metrics(p2,d['y_test'])
        assert abs(measured['accuracy']-r['test']['accuracy'])<1e-14
        assert abs(measured['mse']-r['test']['mse'])<1e-14
        evaluated+=1
    train_keys={r.tobytes() for r in d['raw_train']}
    val_keys={r.tobytes() for r in d['raw_val']}
    test_keys={r.tobytes() for r in d['raw_test']}
    assert not (train_keys & val_keys or train_keys & test_keys or val_keys & test_keys)
    check={'unique_completed_runs':len(rows),'stage_counts':dict(counts),'recomputed_test_runs':evaluated,
           'maximum_prediction_error':max_error,'disjoint_feature_vectors':True}
    (ROOT/'results/release_check.json').write_text(json.dumps(check,indent=2))
    print(json.dumps(check,indent=2))


if __name__=='__main__':main()
