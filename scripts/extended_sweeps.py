"""Run the additional optimizer and learning-rate settings in the manuscript."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from qru.data import ROOT, load_data
from qru.models import QRU
from qru.train import fit


def run(job):
    name, optimizer, rate, seed = job
    data = load_data()
    _, info = fit(QRU(4), data["x_train"], data["y_train"], data["x_val"],
                  data["y_val"], seed=seed, epochs=50, lr=rate, optimizer=optimizer)
    return name, info["trace"], info["best_epoch"]


def summarize(name, optimizer, rate, results):
    selected = [trace[best] for _, trace, best in results]
    row = {"setting": name, "optimizer": optimizer, "rate": rate}
    for metric in ("train_accuracy", "val_accuracy", "val_mse"):
        values = np.array([point[metric] for point in selected])
        row[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    if optimizer == "Adam":
        row["trace"] = {
            metric: np.round(np.mean([[p[metric] for p in trace]
                                      for _, trace, _ in results], axis=0), 9).tolist()
            for metric in ("val_mse", "val_accuracy")
        }
    return row


def main(workers):
    settings = [
        ("Adamax", "Adamax", .003), ("NAdam", "NAdam", .003), ("AdamW", "AdamW", .003),
        ("5e-6", "Adam", 5e-6), ("5e-3", "Adam", 5e-3),
    ]
    jobs = [(*setting, seed) for setting in settings for seed in range(3)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, jobs))
    rows = [summarize(*setting, [r for r in results if r[0] == setting[0]])
            for setting in settings]
    output = ROOT / "results" / "extended_sweeps.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    main(parser.parse_args().workers)
