"""Run the additional optimizer and learning-rate settings in the manuscript."""
import argparse
from concurrent.futures import ProcessPoolExecutor
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
from csv_results import read_rows, write_rows


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


def write_sweeps(rows):
    summaries, curves = [], []
    for row in rows:
        summaries.append({key: value for key, value in row.items() if key != "trace"})
        if "trace" in row:
            trace = row["trace"]
            curves.extend(dict(setting=row["setting"], epoch=i, val_mse=mse, val_accuracy=acc)
                          for i, (mse, acc) in enumerate(zip(trace["val_mse"], trace["val_accuracy"])))
    write_rows(ROOT / "results/extended_sweeps.csv", summaries)
    write_rows(ROOT / "results/extended_curves.csv", curves)


def read_sweeps():
    rows = read_rows(ROOT / "results/extended_sweeps.csv")
    settings = {row["setting"]: row for row in rows}
    for point in read_rows(ROOT / "results/extended_curves.csv"):
        trace = settings[point["setting"]].setdefault("trace", {"val_mse": [], "val_accuracy": []})
        for metric in trace:
            trace[metric].append(point[metric])
    return rows


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
    write_sweeps(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    main(parser.parse_args().workers)
