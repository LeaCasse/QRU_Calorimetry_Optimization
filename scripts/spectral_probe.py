"""Fourier probe over one complete period of one coordinate."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import json
import numpy as np
import pandas as pd
from qru.models import QRU, VQC
from qru.data import ROOT


def main():
    rng = np.random.default_rng(2026); n, draws, depth, eps = 128, 100, 4, .01
    grid = -np.pi+2*np.pi*np.arange(n)/n
    values = {}
    max_leakage = {}
    for name, model, bound in [('QRU_fixed', QRU(depth), depth), ('VQC', VQC(depth), 1)]:
        spectrum = []
        for _ in range(draws):
            p = rng.uniform(-np.pi, np.pi, model.size)
            if name == 'QRU_fixed': p.reshape(-1, 3)[:, 1] = 1.
            x = np.broadcast_to(rng.uniform(-np.pi, np.pi, 3), (n, 3)).copy(); x[:, 0] = grid
            coeff = np.fft.fftshift(np.fft.fft(model.forward(p, x))/n)
            spectrum.append(abs(coeff))
        spectrum = np.asarray(spectrum); orders = np.arange(-n//2, n//2)
        max_leakage[name] = float(spectrum[:, abs(orders)>bound].max())
        assert max_leakage[name] < 1e-12
        values[name+'_activation'] = np.mean(spectrum>eps, axis=0)
        values[name+'_mean_amplitude'] = spectrum.mean(0)
    pd.DataFrame({'order':orders, **values}).query('abs(order) <= 8').to_csv(ROOT/'results/spectrum.csv', index=False)
    (ROOT/'results/spectrum.json').write_text(json.dumps({'draws': draws,'points': n,'depth': depth,
        'threshold': eps,'seed': 2026,'coordinate': 0,'other_coordinates': 'independently uniform [-pi,pi] per parameter draw',
        'trainable_angles': 'uniform [-pi,pi]', 'QRU_encoding_factors': 'fixed at one for this diagnostic only',
        'maximum_outside_support': max_leakage}, indent=2))


if __name__ == '__main__': main()
