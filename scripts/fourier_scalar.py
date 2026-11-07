"""Compare coordinate Fourier order and global parity with one scalar readout."""
import argparse
import itertools
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from qru.data import ROOT, load_data
from qru.models import classify
from csv_results import write_rows


ORDERS = (1, 2)
PENALTIES = np.logspace(-8, -1, 15)


def basis(x, order, even_only):
    """Tensor products of normalized real Fourier functions on the angle cube."""
    columns = []
    for indices in itertools.product(range(2 * order + 1), repeat=x.shape[1]):
        if even_only and sum(i % 2 for i in indices) % 2:
            continue
        column = np.ones(len(x))
        for j, i in enumerate(indices):
            if i:
                trig = np.sin if i % 2 else np.cos
                column *= np.sqrt(2) * trig(((i + 1) // 2) * x[:, j])
        columns.append(column)
    return np.column_stack(columns)


def fit_on_validation(x, y, xv, yv, order, even_only):
    design = basis(x, order, even_only)
    validation = basis(xv, order, even_only)
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    targets = u.T @ y
    weights = vt.T @ (s[:, None] * targets[:, None] /
                      (s[:, None] ** 2 + len(y) * PENALTIES))
    errors = ((validation @ weights - yv[:, None]) ** 2).mean(axis=0)
    best = int(errors.argmin())
    return weights[:, best], float(PENALTIES[best]), errors.tolist()


def metrics(predictions, target):
    return {
        "accuracy": float(np.mean(classify(predictions) == target)),
        "mse": float(np.mean((predictions - target) ** 2)),
    }


def paired_intervals(predictions, target, raw):
    """Conditional percentile intervals from resampling identical-vector groups."""
    pairs = [
        ("order_even", "k2_even", "k1_even"),
        ("order_full", "k2_full", "k1_full"),
        ("parity_k1", "k1_full", "k1_even"),
        ("parity_k2", "k2_full", "k2_even"),
    ]
    _, groups = np.unique(raw, axis=0, return_inverse=True)
    group_sizes = np.bincount(groups)
    differences = np.column_stack([
        np.bincount(groups, weights=(
            (classify(predictions[a]) == target).astype(int) -
            (classify(predictions[b]) == target).astype(int)
        )) for _, a, b in pairs
    ])
    rng = np.random.default_rng(827)
    samples = []
    for _ in range(40):
        indices = rng.integers(len(group_sizes), size=(100, len(group_sizes)))
        samples.append(differences[indices].sum(axis=1) / group_sizes[indices].sum(axis=1)[:, None])
    bounds = np.quantile(np.concatenate(samples), [0.025, 0.975], axis=0)
    return {
        name: {"accuracy_delta": float(differences[:, j].sum() / len(target)),
               "interval": bounds[:, j].tolist()}
        for j, (name, _, _) in enumerate(pairs)
    }


def write_table(rows, directory):
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\caption{Scalar Fourier controls}",
        r"\label{tab:scalar_fourier}", r"\small", r"\begin{tabular}{rlrrrr}",
        r"\toprule",
        r"Order & Parity & Terms & Val. acc. (\%) & Test acc. (\%) & Test MSE \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['order']} & {row['parity']} & {row['terms']} & "
            f"{100 * row['validation']['accuracy']:.2f} & "
            f"{100 * row['test']['accuracy']:.2f} & {row['test']['mse']:.4f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (directory / "fourier_scalar.tex").write_text("\n".join(lines) + "\n")


def write_fourier(rows, penalty_grid, counts, contrasts):
    directory = ROOT / "results"
    write_rows(directory / "fourier_scalar.csv",
               (dict({k: v for k, v in row.items() if k != "validation_mse_grid"},
                     n_train=counts["train"], n_validation=counts["validation"], n_test=counts["test"])
                for row in rows))
    write_rows(directory / "fourier_grid.csv",
               (dict(order=row["order"], parity=row["parity"], penalty=penalty, validation_mse=mse)
                for row in rows for penalty, mse in zip(penalty_grid, row["validation_mse_grid"])))
    write_rows(directory / "fourier_contrasts.csv",
               (dict(contrast=name, accuracy_delta=row["accuracy_delta"],
                     lower=row["interval"][0], upper=row["interval"][1])
                for name, row in contrasts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overleaf", type=Path)
    args = parser.parse_args()
    data = load_data()
    x, y = data["x_train"], data["y_train"]
    xv, yv = data["x_val"], data["y_val"]
    fitted = {}
    for order in ORDERS:
        for even_only in (True, False):
            key = f"k{order}_{'even' if even_only else 'full'}"
            fitted[key] = (order, even_only, *fit_on_validation(x, y, xv, yv, order, even_only))

    xt, yt = data["x_test"], data["y_test"]
    predictions, rows = {}, []
    for key, (order, even_only, weights, penalty, validation_curve) in fitted.items():
        test_design = basis(xt, order, even_only)
        full_terms = (2 * order + 1) ** x.shape[1]
        expected_terms = (full_terms + 1) // 2 if even_only else full_terms
        assert len(weights) == expected_terms
        if even_only:
            assert np.allclose(test_design, basis(-xt, order, even_only), rtol=0, atol=1e-14)
        predictions[key] = test_design @ weights
        rows.append({
            "order": order,
            "parity": "Even" if even_only else "Unrestricted",
            "terms": len(weights),
            "penalty": penalty,
            "validation_mse_grid": validation_curve,
            "validation": metrics(basis(xv, order, even_only) @ weights, yv),
            "test": metrics(predictions[key], yt),
        })
    counts = {"train": len(y), "validation": len(yv), "test": len(yt)}
    write_fourier(rows, PENALTIES, counts, paired_intervals(predictions, yt, data["raw_test"]))
    np.savez_compressed(ROOT / "results/fourier_scalar_predictions.npz", **predictions)
    write_table(rows, ROOT / "results/tables")
    if args.overleaf:
        write_table(rows, args.overleaf / "tables")
    print((ROOT / "results/fourier_scalar.csv").read_text().rstrip())


if __name__ == "__main__":
    main()
