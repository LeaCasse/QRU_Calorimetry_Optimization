# QRU calorimetry

Here is the code for the revised paper. The small NumPy simulators calculate the same circuit outputs and gradients without the old per-event framework overhead.

Use Python 3.12, then run:

```bash
pip install -r requirements.txt
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

Saved results are in CSV files. `results/runs.csv` lists the runs and `results/training_curves.csv.gz` holds their epoch-by-epoch metrics. The `.npz` files contain arrays used by the reproduction scripts.

The IQM Garnet figure in the manuscript reports a separate experiment with a seven-layer, single-qubit circuit and ten inputs, using 50 and 200 shots. The measured scores gave the same class decisions as the simulator for these inputs. The same evaluation loader was used for checkpoint selection. The hardware figure is in the manuscript and the raw device counts and checkpoint are unavailable. `scripts/export_inference.py` exports QASM and CSV references for a saved benchmark run.
