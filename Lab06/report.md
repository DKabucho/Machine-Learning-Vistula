# Lab 6 — Reproducible Fashion-MNIST Training

## Reproducibility and model

The experiments reuse the Lab 3 CNN (20,490 parameters), Fashion-MNIST
normalization (`mean=0.2860`, `std=0.3530`), and its reproducible 55,000/5,000
training/validation split. The official 10,000-image test split is kept separate.
All run parameters, including seed, loader settings, optimizer, scheduler,
checkpoint/output locations, and MLflow tracking directory, are declared in
[`config.yaml`](./config.yaml). `src/train.py` reads that YAML mapping once and
uses the same training/evaluation implementation for the requested comparisons.

Run commands from the `Lab06` directory with the repository virtual environment:

```powershell
..\.venv\Scripts\python.exe src\train.py train
..\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri results\mlruns
```

The default five-epoch run logs configuration, per-epoch training/validation
loss and accuracy, learning rate, final test metrics, YAML, history CSV, and
checkpoint as MLflow params, metrics, and artifacts. Checkpoints include model,
optimizer, scheduler, history, and Python/NumPy/PyTorch RNG states. Results,
checkpoints, traces, and MLflow's local store are generated under `results/`.
The Fashion-MNIST files are reused from `../Lab03/data`; set `download: true`
in the YAML only if that local copy is unavailable.

## Profiling and data-loader workers

Run:

```powershell
..\.venv\Scripts\python.exe src\train.py profile
..\.venv\Scripts\python.exe src\train.py workers
```

Stage timing covers a full training epoch. The PyTorch profiler captures a
scheduled representative window (one warm-up and five active batches) in its
Chrome trace to keep the trace manageable; stage shares remain full-epoch
wall-clock times for data retrieval, forward pass, backward pass, and optimizer
step. CUDA is synchronized at stage boundaries if available. The command writes
the trace, `results/profile.json`, and `results/profile.csv`. The worker sweep uses
`num_workers = 0, 1, 2, 4, 8`, one epoch per setting, and writes
`results/worker-sweep.csv` and `results/worker-sweep.png`. Worker timing is
end-to-end train-plus-validation duration; the training and validation
durations are also recorded separately in the history CSV. Validation uses
zero workers for every run. The lowest epoch time identifies the best tested
worker setting; a rise at larger counts indicates worker
startup/coordination overhead rather than more useful data-loading parallelism.

| Profile stage | Seconds | Share |
|---|---:|---:|
| Data loading | 32.80 | 62.24% |
| Forward | 8.05 | 15.27% |
| Backward | 10.44 | 19.81% |
| Optimizer step | 1.41 | 2.68% |

On this CPU-only run (PyTorch 2.3.1), data retrieval dominated at 62.24% of
the profiled stage time, making data loading the immediate bottleneck. This
measurement includes obtaining the batch and moving it to the selected device;
the Chrome trace contains the scheduled representative window, while the
stage percentages use the complete epoch.

| Workers | 0 | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| Epoch seconds | 67.16 | 69.92 | 82.08 | 145.79 | 247.66 |

With this CPU and in-memory dataset, zero workers was fastest. Additional
processes increasingly hurt throughput, with eight workers taking about 3.7×
the zero-worker training time; this run therefore does not indicate a useful
data-loading parallelism bottleneck that worker processes can solve.

![Training time by data-loader worker count](./results/worker-sweep.png)

## Batch-size study

Run `..\.venv\Scripts\python.exe src\train.py batch-sizes`. Each batch size
`[32, 64, 128, 256, 512]` gets a fresh model with the same seed and five epochs.
Throughput is training examples processed per training second (validation time
is excluded); peak memory is CUDA peak allocated memory, or the maximum process
RSS sampled at batch boundaries on CPU.
Final accuracy is measured on the official test set. Results and the throughput
plot are saved as `results/batch-size-sweep.csv` and
`results/batch-size-sweep.png`.

| Batch | Throughput (samples/s) | Peak memory (MB) | Test accuracy |
|---:|---:|---:|---:|
| 32 | _See results/batch-size-sweep.csv_ | | |
| 64 | | | |
| 128 | | | |
| 256 | | | |
| 512 | | | |

Larger batches can improve device utilization and throughput but require more
activation memory. Results are hardware-dependent; no batch-size winner is
assumed before running the measurements. This five-epoch sweep was not run in
the current CPU session; its table will fill from the command output.

## Optimizers, schedulers, and gradient accumulation

Run `..\.venv\Scripts\python.exe src\train.py optimizers` for the six
SGD-with-momentum/Adam × StepLR/CosineAnnealingLR/ReduceLROnPlateau
combinations. Every run uses a ten-epoch budget and seed 42. The output
`results/optimizer-scheduler.csv` includes final test accuracy and the associated
history CSVs retain the training and validation curves.

Run `..\.venv\Scripts\python.exe src\train.py accumulation` to compare a batch
of 256 with micro-batches of 64 accumulated over four steps, for five epochs.
The summed per-example gradients are normalized by the number of examples in
each accumulation window, including the final partial window. This matches the
effective batch gradient (up to floating-point ordering), while requiring less
activation memory. `results/gradient-accumulation.csv` records the comparison.
The ten-epoch optimizer/scheduler matrix and five-epoch accumulation comparison
are implemented but not run in this CPU session; their numeric comparisons
remain pending generated results.

| Configuration | Test accuracy | Observation |
|---|---:|---|
| Batch 256, accumulation 1 | _See results/gradient-accumulation.csv_ | |
| Batch 64, accumulation 4 | | |

## Checkpoint resume

Stop the configured five-epoch run after epoch three, then resume to epoch five:

```powershell
..\.venv\Scripts\python.exe src\train.py train --stop-after 3
..\.venv\Scripts\python.exe src\train.py train --resume results\checkpoints\cnn-training-last.pt
```

The resumed history CSV contains epochs 1–5. Compare its first three entries
with the initial run and inspect train/validation loss and accuracy versus
epoch; optimizer/scheduler and RNG states are restored from the epoch-3
checkpoint. In the verified two-epoch smoke run, train loss fell from 0.5295
to 0.3511 and validation accuracy rose from 84.66% to 87.08%. Each invocation is
separately recorded in MLflow.

## Individual project proposal

**Student:** Daniel Kabucho Karanja

**Student ID:** 70470

**Problem.** Build a used-car price-estimation system for a dealership that
accepts structured listing attributes and returns a suggested price for staff
review. The project will compare interpretable, tree-based, and neural
tabular approaches, select a deployment-feasible model, and measure the
complete preprocessing-and-inference pipeline.

**Dataset.** The Kaggle
[Craigslist Used Cars](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
dataset (`austinreese/craigslist-carstrucks-data`), approximately 426,000
vehicle listings with a `vehicles.csv` file. Candidate features include year,
manufacturer, model, condition, odometer, fuel, transmission, drive, type,
paint color, and state. Records outside a documented price range or with
critical missing fields will be excluded or imputed according to rules fitted
on training data only. The alternative-dataset approval and source, license,
size, and bias documentation are required before implementation is submitted.

**Success metric.** Mean absolute error (MAE) no greater than $3,000 on a
held-out test partition. Final reporting will also include median absolute
error, RMSE, R-squared, and deployment pass/fail flags. These targets remain
pending measured results because the Kaggle CSV is not currently stored in the
repository.

**Deployment constraints.** End-to-end prediction latency no greater than
50 ms per vehicle and serialized model plus preprocessing pipeline no greater
than 5 MB.

**Integrated implementation.** The project combines concepts from Labs 1–6:

* **Lab 1 system measurement:** warm up every benchmark, repeat training and
  100 single-request inference trials, report medians, and measure peak RSS,
  throughput, latency, and serialized size against the stated limits.
* **Lab 2 neural-network foundations:** a compact from-scratch or PyTorch
  tabular MLP using forward pass, backpropagation, mini-batch training,
  learning-rate comparisons, and gradient checks where practical.
* **Lab 3 architecture trade-offs:** compare Ridge, histogram gradient
  boosting, and a two-hidden-layer MLP with documented parameter counts.
  CNN and attention models are unsuitable for structured data unless listing
  text or images are added as new inputs.
* **Lab 4 data quality:** define schema and range checks, record violations,
  split 60/20/20 before fitting transformations, clip numeric outliers using
  training percentiles, deduplicate the training partition, and retain the test
  partition unchanged. Derived features such as vehicle age and mileage per
  year will be evaluated. The run will produce a data card, processed-data
  SHA-256 hash, validation report, and fitted transformer.
* **Lab 5 framework analysis:** validate MLP gradients against PyTorch
  autograd and numerical estimates, inspect eager and traced or compiled
  execution, and report tensor shape, dtype, contiguity, tracing, and
  optimization overhead.
* **Lab 6 training pipeline:** make settings configurable in YAML, log params
  and metrics to MLflow, profile one epoch, compare data-loader workers,
  batch sizes, optimizer/scheduler combinations, gradient accumulation, and
  checkpoint resume, and save curves and artifacts.

**Model selection.** Ridge and histogram gradient boosting provide
interpretable and non-linear classical baselines. A tabular MLP provides a
neural comparison. Preprocessing will be fitted inside a leakage-free pipeline.
Candidates will be compared on validation MAE; the winner will be retrained on
training plus validation data and tested once. The selected model must meet
accuracy, size, and latency targets or undergo quantization, pruning,
distillation, feature reduction, or smaller-architecture search.

**Optimization plan.** First evaluate post-training quantization or a
quantized ONNX Runtime variant for the MLP. If accuracy remains acceptable,
report quantized size and p95 latency. If the classical model is smaller and
faster, consider feature pruning or categorical-vocabulary compression.
Distillation is a fallback: train a compact student from the best larger model
and compare deployment metrics.

The implementation currently provides leakage-safe cleaning, deterministic
60/20/20 splitting, Ridge and histogram-gradient-boosting pipelines, unit
tests, a data card, and a reproducible run command. It is extended in scope by
the MLP, MLflow, profiling, batch/worker/optimizer studies, checkpoint resume,
and compression stages described above. Raw data, generated models, the MLflow
store, and measured result files remain excluded from Git.
