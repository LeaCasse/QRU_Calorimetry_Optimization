# QRU calorimetry

Here is the code for the revised paper. The small NumPy simulators calculate the same circuit outputs and gradients without the old per-event framework overhead.

Use Python 3.12, then run:

```bash
pip install -r requirements.txt
python scripts/build_outputs.py
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

The manuscript's IQM Garnet figure shows a separate hardware execution. [Execution details](results/qpu_execution.md) document its protocol and how it relates to the numerical benchmark.
