# Individual Project: Compact Fashion-MNIST Classifier

**Student:** Daniel Kabucho Karanja  
**Student ID:** 70470

## Project goal

Evaluate whether dynamic int8 quantization can make the Lab 3 Fashion-MNIST
CNN more deployment-friendly while preserving at least 88% test accuracy,
keeping batch-1 p95 inference latency at or below 50 ms, and keeping serialized
weights at or below 5 MB.

## Baseline and dataset

The project reuses the Lab 3 CNN, Fashion-MNIST normalization, and official
10,000-image test set. The five-epoch baseline checkpoint is produced by the
Lab 6 YAML-driven training pipeline. The comparison script evaluates the
float32 checkpoint and a dynamically quantized copy on the same test examples.
Latency is measured on CPU after warm-up; reported model latency excludes image
decoding and normalization, which will need separate measurement if those
steps are part of the final deployment path.

## First experiment

After training the baseline, run from the repository root:

```powershell
& '.\.venv\Scripts\python.exe' '.\IndividualProject\src\compare_quantization.py'
```

The script writes `IndividualProject/results/quantization-comparison.csv` and
`.json`, with test accuracy, batch-1 median and p95 latency, serialized
state-dict size, and pass/fail for each target. Generated outputs are ignored
by Git; the measured values from this run are recorded below.

| Model | Test accuracy | Median / p95 latency (ms) | State-dict size (MB) | Meets all targets |
|---|---:|---:|---:|---|
| Float32 CNN | 89.22% | 0.840 / 1.462 | 0.0805 | Yes |
| Dynamic int8 CNN | 89.19% | 1.094 / 1.813 | 0.0364 | Yes |

On this CPU, dynamic int8 reduced serialized state-dict size by about 54.8%
and lowered test accuracy by 0.03 percentage points. It was slower at batch 1
(p95 increased by 0.351 ms), though both versions are comfortably within the
50 ms limit. Since the float model already meets the size and latency budgets,
quantization is a viable smaller variant but not a latency optimization for
this CNN/hardware combination. These are CPU model-only latency measurements;
image decoding and preprocessing are excluded.

## Current status

The five-epoch baseline was trained on CPU. Validation accuracy rose from
84.66% after epoch 1 to 89.84% after epoch 5. Both float and quantized models
exceeded the 88% test accuracy target and met the stated size and latency
budgets on this CPU. Further work should measure preprocessing-inclusive
latency and test deployment on the intended target device before choosing the
quantized model; if latency remains the priority, investigate a more suitable
optimization rather than assuming quantization is faster.
