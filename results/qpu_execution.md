# IQM Garnet inference

The hardware figure in the manuscript records a seven-layer, three-input QRU run through qBraid on IQM Garnet. The single-qubit QRU was trained with PennyLane's statevector simulator using MSE and Adadelta at learning rate 0.5. The fitted parameters were frozen. An OpenQASM 2.0 circuit was formed for each event, and the score was reconstructed as the difference of the two computational-basis counts divided by their sum. The decisions use thresholds of −0.33 and +0.33.

The exact simulator and both device runs (50 and 200 shots per event) classify all ten events correctly. These are the first ten entries of the evaluation loader, which was also used for checkpoint selection in the training script for this run. The hardware comparison checks execution agreement on those inputs. The full-partition results and the ideal finite-shot calculation use separate checkpoints with validation-based selection.

The event-level output figure is retained with the manuscript. Raw device counts, the frozen checkpoint and job metadata are unavailable in this repository. The current `export_inference.py` utility exports a chosen benchmark checkpoint and its exact predictions for a new execution.
