"""Load the released arrays and make a training-only preprocessing split."""
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]


def load_data(seed=2026):
    a = np.load(ROOT / 'data/calorimetry.npz', allow_pickle=False)
    raw, target = a['raw_train'].astype(float), a['y_train'].astype(int)
    test, yt = a['raw_test'].astype(float), a['y_test'].astype(int)
    # Identical feature vectors are kept in the same partition.
    test_keys = {r.tobytes() for r in test}
    keep = np.array([r.tobytes() not in test_keys for r in raw])
    raw, target = raw[keep], target[keep]
    groups, first, inverse = np.unique(raw, axis=0, return_index=True, return_inverse=True)
    gt = target[first]
    for i in np.flatnonzero(np.bincount(inverse) > 1):
        if len(np.unique(target[inverse == i])) != 1:
            raise ValueError('Conflicting labels for an identical feature vector')
    tr_groups, va_groups = train_test_split(np.arange(len(groups)), test_size=.2,
                                           random_state=seed, stratify=gt)
    ti, vi = np.flatnonzero(np.isin(inverse, tr_groups)), np.flatnonzero(np.isin(inverse, va_groups))
    lo, hi = raw[ti].min(0), raw[ti].max(0)
    def scale(x, mode='centered'):
        if mode == 'standard':
            return (x-raw[ti].mean(0))/raw[ti].std(0)
        u = (x-lo)/np.maximum(hi-lo, 1e-12)
        return u*2*np.pi if mode == 'positive' else u*2*np.pi-np.pi
    return {'raw_train': raw[ti], 'raw_val': raw[vi], 'raw_test': test,
            'x_train': scale(raw[ti]), 'x_val': scale(raw[vi]), 'x_test': scale(test),
            'y_train': target[ti], 'y_val': target[vi], 'y_test': yt,
            'train_indices': np.flatnonzero(keep)[ti], 'val_indices': np.flatnonzero(keep)[vi],
            'scale': scale, 'minimum': lo, 'maximum': hi,
            'removed_overlap': int((~keep).sum())}
