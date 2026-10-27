"""Export a frozen revised QRU to OpenQASM 2. This script submits no jobs."""
import argparse
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from qru.data import load_data, ROOT
from qru.train import make_model
from qru.models import classify


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
    cached = ROOT/'results/comparison'/f'{args.run_id}.json'
    if cached.exists():
        row = json.loads(cached.read_text())
        theta = np.load(ROOT/'results/comparison'/f'{args.run_id}.npz')['theta']
    else:
        import gzip
        with gzip.open(ROOT/'results/runs.json.gz','rt') as f:
            row = next(r for r in json.load(f) if r['id']==args.run_id)
        theta = np.load(ROOT/'results/models.npz')[args.run_id+'_theta']
    config = row['config']
    if config['kind'] != 'QRU': raise ValueError('Select a QRU run')
    d = load_data(); model = make_model(**config)
    x = d['scale'](d['raw_test'], config.get('normalization','centered'))[:args.count]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for i, value in enumerate(x): (out/f'event_{i:04d}.qasm').write_text(qasm(model, theta, value))
    ref = model.forward(theta, x)
    (out/'reference.json').write_text(json.dumps({'run_id':args.run_id,'config':config,
        'expectation':ref.tolist(),'prediction':classify(ref).tolist(),'label':d['y_test'][:args.count].tolist(),
        'input':x.tolist(),'theta':theta.tolist(),'hardware_submitted':False},indent=2))
