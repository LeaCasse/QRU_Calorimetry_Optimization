"""Export a frozen revised QRU to OpenQASM 2. This script submits no jobs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from qru.data import load_data, ROOT
from qru.train import make_model
from qru.models import classify
from csv_results import cache_paths, read_runs, write_rows


def qasm(model, theta, x):
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[1];', 'creg c[1];']
    for axis, angle, _ in model.gates(theta, x[None, :]):
        value = float(np.asarray(angle).ravel()[0])
        lines.append(f'r{"xyz"[axis]}({value:.17g}) q[0];')
    return '\n'.join(lines+['measure q[0] -> c[0];', ''])


def from_counts(counts):
    cleaned = {str(k).replace(' ', ''): int(v) for k, v in counts.items()}
    if set(cleaned)-{'0','1'} or any(v < 0 for v in cleaned.values()) or sum(cleaned.values()) <= 0:
        raise ValueError('Expected nonnegative single-qubit counts for outcomes 0 and 1')
    return (cleaned.get('0',0)-cleaned.get('1',0))/sum(cleaned.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run_id'); parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--output', default='inference_export'); args = parser.parse_args()
    cached, curve = cache_paths('comparison', args.run_id)
    if cached.exists():
        row = read_runs(cached, curve)[0]
        with np.load(cached.with_suffix('.npz')) as arrays:
            theta = arrays['theta']
    else:
        row = next(r for r in read_runs() if r['id']==args.run_id)
        with np.load(ROOT/'results/models.npz') as arrays:
            theta = arrays[args.run_id+'_theta']
    config = row['config']
    if config['kind'] != 'QRU': raise ValueError('Select a QRU run')
    d = load_data(); model = make_model(**config)
    x = d['scale'](d['raw_test'], config.get('normalization','centered'))[:args.count]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for i, value in enumerate(x): (out/f'event_{i:04d}.qasm').write_text(qasm(model, theta, value))
    ref = model.forward(theta, x)
    write_rows(out/'reference.csv',
               (dict(event=i, x1=features[0], x2=features[1], x3=features[2],
                     expectation=score, prediction=label, target=target)
                for i,(features,score,label,target) in
                enumerate(zip(x,ref,classify(ref),d['y_test'][:args.count]))))
    write_rows(out/'parameters.csv',
               (dict(parameter=i, value=value) for i,value in enumerate(theta)))
    write_rows(out/'configuration.csv',
               [dict(run_id=args.run_id, hardware_submitted=False, **config)])
