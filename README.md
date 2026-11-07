# QRU calorimetry

Here is the code for the revised paper. The small NumPy simulators calculate the same circuit outputs and gradients without the old per-event framework overhead.

Use Python 3.12, then run:

```bash
pip install -r requirements.txt
python scripts/check_models.py
python scripts/check_release.py
python scripts/build_outputs.py
python scripts/build_paper_figures.py
```

The released results are already included. To train everything again:

```bash
python scripts/run_experiments.py tune --recompute
python scripts/run_experiments.py comparison --recompute
python scripts/run_experiments.py sweeps --recompute
python scripts/run_experiments.py features --recompute
python scripts/run_experiments.py search --recompute
python scripts/classical_controls.py
python scripts/benchmark_runtime.py
python scripts/benchmark_runtime.py --depth-sweep
python scripts/check_initializations.py
python scripts/extended_sweeps.py
python scripts/spectral_probe.py
python scripts/fourier_scalar.py
python scripts/build_outputs.py
python scripts/build_paper_figures.py
```

Runs save as they finish. Omit `--recompute` to resume. `results/runs.json.gz` holds configurations and trajectories, `models.npz` holds weights and predictions, and `figures` and `tables` are generated from those numbers.

The [numeric benchmark](data/events.csv.gz) contains the exact data used here. Its column names are preserved. Physical units and the mapping from class codes to particle names are unverified. See `data/provenance.json` for the source hashes.

`export_inference.py RUN_ID` writes QASM and exact references. It submits no hardware jobs. Earlier scripts and logs remain in Git history.

To rebuild the numeric release from the pinned original files, run `python scripts/prepare_data.py --download`. The parser reads tensor bytes without executing pickle objects and checks source hashes.

The scalar Fourier control fits four models with the same class targets and thresholds as the QRU. Run `python scripts/fourier_scalar.py --overleaf PATH` to reproduce its scores, paired accuracy intervals and manuscript table.

The depth timing sweep measures 50 epochs at each depth on one numerical thread. The initialization check varies the starting parameters of the four-layer QRU while holding the data split and minibatch order fixed. Both scripts write their numerical results under `results/` for `build_outputs.py`.

The manuscript's IQM Garnet figure shows a separate hardware execution. [Execution details](results/qpu_execution.md) document its protocol and how it relates to the numerical benchmark.

To refresh the figures and tables in an extracted Overleaf project, run
`python scripts/build_outputs.py --overleaf PATH` followed by
`python scripts/build_paper_figures.py --overleaf PATH`. The second command
generates the detailed plots and tables for `main_V4.tex`, including the
optimizer and learning-rate settings in `results/extended_sweeps.json`.
