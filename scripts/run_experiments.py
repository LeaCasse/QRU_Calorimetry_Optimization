"""Run the revision experiments. Every completed run is saved immediately."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import itertools
import json
import sys
import time
from functools import lru_cache
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from qru.data import ROOT, load_data
from qru.models import classify
from qru.train import make_model, fit, metrics


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)


@lru_cache(maxsize=1)
def released_runs():
    import gzip
    path=ROOT/'results/runs.json.gz'
    if not path.exists():return {}
    with gzip.open(path,'rt') as f:return {r['id']:r for r in json.load(f)}


def run_one(config):
    name = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:14]
    stage = config['stage']; path = ROOT / 'results' / stage / (name+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not os.environ.get('QRU_RECOMPUTE'):
        return json.loads(path.read_text())
    if not os.environ.get('QRU_RECOMPUTE') and name in released_runs():
        return released_runs()[name]
    d = load_data(); mode = config.get('normalization', 'centered')
    x, xv = d['scale'](d['raw_train'], mode), d['scale'](d['raw_val'], mode)
    y, yv = d['y_train'], d['y_val']
    features = config.get('features', [0, 1, 2]); x, xv = x[:, features], xv[:, features]
    subset = config.get('subset')
    if subset and subset < len(x):
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(np.arange(len(x)), train_size=subset, random_state=111, stratify=y)
        x, y = x[idx], y[idx]
    model = make_model(dimensions=len(features), **config)
    epochs = config.get('epochs', 50); batch = config.get('batch_size', 256)
    if batch == 'full':
        batch = len(x)
    theta, info = fit(model, x, y, xv, yv, seed=config.get('seed', 0), epochs=epochs,
                      batch_size=batch, lr=config.get('lr', .003),
                      optimizer=config.get('optimizer', 'Adam'), loss=config.get('loss', 'L2'),
                      schedule=config.get('schedule', False))
    pred_val = model.forward(theta, xv)
    result = {'id': name, 'config': config, 'fit': info, 'validation': metrics(pred_val, yv),
              'training': metrics(model.forward(theta, x), y), 'n_train': len(x), 'n_val': len(xv)}
    weights = ROOT / 'results' / stage / (name+'.npz')
    if stage in ['comparison', 'feature_ablation']:
        xt = d['scale'](d['raw_test'], mode)[:, features]
        pred = model.forward(theta, xt)
        result['test'] = metrics(pred, d['y_test'])
        np.savez_compressed(weights, theta=theta, prediction=pred, validation_prediction=pred_val)
    else:
        np.savez_compressed(weights, theta=theta)
    save_json(path, result)
    return result


def execute(configs, workers=3):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, c) for c in configs]
        for f in as_completed(futures):
            r = f.result(); c = r['config']
            print(c['stage'], c.get('group', ''), c.get('kind', 'QRU'), c.get('seed', 0),
                  round(r['validation']['accuracy'], 4), r['id'], flush=True)


def sweep_configs():
    base = {'stage': 'sweeps', 'epochs': 50}
    jobs = []
    def add(group, variants, **fixed):
        for variant in variants:
            for seed in range(3):
                jobs.append(dict(base, group=group, seed=seed, **fixed, **variant))
    add('depth', [{'depth': x} for x in [1, 2, 3, 4, 6, 10, 14]])
    add('normalization', [{'normalization': x} for x in ['centered', 'positive', 'standard']], depth=4)
    add('axes', [{'axes': x} for x in ['XYX', 'XZX', 'YXY', 'YZY', 'ZXZ', 'ZYZ', 'XYZ']], depth=4)
    add('parameters', [{'parameters': x} for x in [1, 2, 3, 4, 5]], depth=4)
    add('loss', [{'loss': x} for x in ['L1', 'L2', 'Huber']], depth=4)
    add('learning_rate', [{'lr': x} for x in [5e-7, 5e-5, 5e-4, .003, .03, .5]], depth=4)
    add('optimizer', [{'optimizer': a, 'lr': b} for a, b in [('SGD', .03), ('RMSprop', .003),
        ('Adam', .003), ('Adagrad', .03), ('Adadelta', 1.)]], depth=4)
    add('schedule', [{'optimizer': a, 'lr': b, 'schedule': s}
        for a, b in [('Adam', .003), ('SGD', .03)] for s in [False, True]], depth=4)
    for batch in [1, 5, 20, 100, 256, 'full']:
        for seed in range(3):
            jobs.append({'stage': 'sweeps', 'group': 'batch', 'depth': 4, 'batch_size': batch,
                         'subset': 4096, 'epochs': 10, 'seed': seed})
    return jobs


def tune():
    configs = []
    for kind, depth, hidden in [('QRU', 4, 17), ('QRU', 10, 17), ('VQC', 10, 17), ('MLP', 10, 17)]:
        for lr in [.0005, .003, .03]:
            configs.append({'stage': 'tuning', 'kind': kind, 'depth': depth, 'hidden': hidden,
                            'lr': lr, 'seed': 123, 'epochs': 50})
    execute(configs)
    selected = {}
    for kind, depth in [('QRU', 4), ('QRU', 10), ('VQC', 10), ('MLP', 10)]:
        rows = [run_one(c) for c in configs if c['kind'] == kind and c['depth'] == depth]
        best = min(rows, key=lambda r: r['validation']['mse'])
        selected[kind+str(depth)] = best['config']
    save_json(ROOT / 'results/selected.json', selected)


def comparison():
    selected = json.loads((ROOT / 'results/selected.json').read_text())
    configs = [dict(c, stage='comparison', seed=seed) for c in selected.values() for seed in range(10)]
    execute(configs)


def features():
    configs = [{'stage': 'feature_ablation', 'features': list(f), 'depth': 4, 'epochs': 50, 'seed': seed}
               for k in [1, 2, 3] for f in itertools.combinations(range(3), k) for seed in range(3)]
    execute(configs)


def search():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from sklearn.preprocessing import OneHotEncoder
    depths, lrs = [1, 2, 3, 4, 6, 10], [.0005, .003, .03, .3, 1.]
    losses, opts = ['L1', 'L2', 'Huber'], ['Adam', 'Adagrad', 'Adadelta', 'SGD']
    grid = list(itertools.product(depths, lrs, losses, opts))
    cats = OneHotEncoder(sparse_output=False).fit_transform(np.array([[c[2], c[3]] for c in grid]))
    xx = np.c_[[c[0]/10 for c in grid], np.log10([c[1] for c in grid])/4, cats]
    rng = np.random.default_rng(2026); used, scores, rows = [], [], []
    for trial in range(50):
        left = np.setdiff1d(np.arange(len(grid)), used)
        if trial < 10:
            idx = int(rng.choice(left))
        else:
            gp = GaussianProcessRegressor(kernel=Matern(length_scale=1., nu=2.5)+WhiteKernel(1e-5),
                                          alpha=1e-6, normalize_y=True, optimizer=None)
            gp.fit(xx[used], scores); mu, sd = gp.predict(xx[left], return_std=True)
            idx = int(left[np.argmin(mu-4*sd)])
        dep, lr, loss, opt = grid[idx]
        c = {'stage': 'bayesian', 'depth': dep, 'lr': lr, 'loss': loss, 'optimizer': opt,
             'epochs': 12, 'subset': 8192, 'seed': 2026}
        r = run_one(c); used.append(idx); scores.append(r['validation']['mse'])
        rows.append({'trial': trial+1, 'grid_index': idx, 'result': r})
        save_json(ROOT/'results/bayesian_search.json', rows)
        print('GP', trial+1, round(scores[-1], 5), c, flush=True)
    # Canonical Hyperband brackets with eta=3 and R=27. Promotions replay the
    # same deterministic trajectory from epoch zero. All replay costs are kept.
    hb = []
    for s in reversed(range(4)):
        n = int(np.ceil(4/(s+1)*3**s)); resource = int(27*3**(-s))
        candidates = list(rng.choice(len(grid), n, replace=False))
        for rung in range(s+1):
            evaluated = []
            for idx in candidates:
                dep, lr, loss, opt = grid[idx]
                c = {'stage': 'hyperband', 'depth': dep, 'lr': lr, 'loss': loss, 'optimizer': opt,
                     'epochs': resource*3**rung, 'subset': 8192, 'seed': 2026}
                r = run_one(c); evaluated.append((r['validation']['mse'], int(idx)))
                hb.append({'bracket': s, 'rung': rung, 'grid_index': int(idx), 'result': r})
                save_json(ROOT/'results/hyperband_search.json', hb)
            candidates = [i for _, i in sorted(evaluated)[:max(1, len(candidates)//3)]]
            print('HB', s, rung, len(evaluated), min(evaluated)[0], flush=True)
    # First-order functional variance of a random-forest surrogate on this
    # finite Cartesian grid. This describes the searched space only.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=2026, n_jobs=1)
    cv = cross_val_score(rf, xx[used], scores, cv=5, scoring='r2')
    rf.fit(xx[used], scores); pred = rf.predict(xx); variance = np.var(pred)
    importance = {}
    for j, name in enumerate(['depth', 'learning_rate', 'loss', 'optimizer']):
        vals = list(dict.fromkeys(c[j] for c in grid))
        marg = [np.mean(pred[[c[j] == v for c in grid]]) for v in vals]
        importance[name] = float(np.var(marg)/variance)
    save_json(ROOT/'results/search_summary.json', {'bayesian_best_mse': min(rows, key=lambda r:r['result']['validation']['mse']),
        'bayesian_best_accuracy': max(rows, key=lambda r:r['result']['validation']['accuracy']),
        'hyperband_best_full_budget': min([r for r in hb if r['result']['config']['epochs'] == 27], key=lambda r:r['result']['validation']['mse']),
        'first_order_importance': importance, 'surrogate_cv_r2': cv.tolist(),
        'bayesian_requested_epochs': sum(r['result']['config']['epochs'] for r in rows),
        'hyperband_requested_epochs': sum(r['result']['config']['epochs'] for r in hb)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['tune', 'comparison', 'sweeps', 'features', 'search'])
    parser.add_argument('--recompute',action='store_true',help='Train again instead of resuming released results')
    args = parser.parse_args()
    if args.recompute:os.environ['QRU_RECOMPUTE']='1'
    if args.stage == 'sweeps': execute(sweep_configs())
    else: globals()[args.stage]()
