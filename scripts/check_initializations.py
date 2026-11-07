"""Evaluate the four-layer QRU across 50 initializations with one batch order."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qru.data import ROOT, load_data
from qru.models import QRU
from qru.train import fit, metrics
from csv_results import write_rows


ORDER_SEED = 2026


def run(seed):
    data = load_data()
    model = QRU(4)
    params, info = fit(
        model, data["x_train"], data["y_train"], data["x_val"], data["y_val"],
        initial=model.initial(seed), seed=ORDER_SEED, epochs=50, lr=.003,
    )
    return {
        "initialization_seed": seed,
        "checkpoint_epoch": info["best_epoch"],
        "validation": metrics(model.forward(params, data["x_val"]), data["y_val"]),
        "test": metrics(model.forward(params, data["x_test"]), data["y_test"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run, range(50)):
            rows.append(row)
            print(row, flush=True)
    settings = {"depth": 4, "epochs": 50, "batch_size": 256, "learning_rate": .003,
                "optimizer": "Adam", "training_loss": "L2", "batch_order_seed": ORDER_SEED}
    write_rows(ROOT / "results/initializations.csv", (dict(row, **settings) for row in rows))
