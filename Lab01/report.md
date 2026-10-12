# Lab 1: Environment and first system measurements

## Goal

Verify the isolated Python environment, establish reproducible linear and
ensemble baselines on the Breast Cancer Wisconsin dataset, and compare their
measured system costs with the Cloud, Edge, Mobile, and TinyML budgets.

## Method

The experiment ran with Python 3.11.8 in the repository's isolated `.venv`.
`results/versions.txt` records the Python version and every package in the
pinned requirements file. The MLflow 2.14.1 pin requires `pyarrow<16`, so the
compatible `pyarrow==15.0.2` pin is used instead of 16.1.0.

The dataset is loaded with scikit-learn's
`load_breast_cancer(return_X_y=True)`; it is included with scikit-learn and
requires no separate download. The data is split once into 70% training and
30% test data with stratification and `random_state=42`. Python, NumPy, and
PyTorch CPU/CUDA random seeds are set to 42. Each requested model is trained
with its specified parameters, given one unmeasured warm-up fit, then fitted
five more times. The last measured fit supplies the test accuracy. Inference
is warmed up once, then timed for 100 individual predictions on one test
sample; the reported latency is their median.

Training time is the median of the five measured fits. Process RSS is sampled
with psutil at 1 ms intervals during each measured fit and during the
100-prediction inference batch. The training memory figure is the median peak
RSS across the five fits; the inference figure is the peak RSS observed over
the prediction batch. Both are absolute process RSS, including Python and
loaded libraries, not just model-attributable memory. Each fitted estimator
is saved separately with joblib, and `results/model.joblib` contains a
joblib bundle of both estimators.

## Results

Test accuracy is rounded to four decimal places in
[`results/baseline_accuracy.csv`](./results/baseline_accuracy.csv). Cost
measurements are also available in
[`results/system_measurements.csv`](./results/system_measurements.csv).

| Model | Test accuracy | Training median (s) | Training peak RSS median (MB) | Single-sample inference median (ms) | Inference peak RSS (MB) | Model size (bytes) | Model size (KB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.9415 | 0.444112 | 230.309 | 0.072450 | 230.324 | 1,055 | 1.030 |
| RandomForestClassifier | 0.9357 | 0.170823 | 231.559 | 3.881650 | 231.648 | 290,889 | 284.071 |

The required combined artifact, `results/model.joblib`, is 291,642 bytes
(284.807 KB). The table reports the size of each model's separate joblib
artifact, so its size can be compared directly with the per-model deployment
budgets.

### Deployment budget comparison

Memory fit is based on the measured inference process RSS. The Cloud, Edge,
and Mobile decisions assume a device meeting the stated memory capacity;
Mobile fit is consequently limited to devices near the 256 MB end of that
range. These RSS values include the full benchmark process and provide a
conservative check rather than an estimate of a stripped-down inference
runtime.

| Model | Target | Memory budget; measured inference RSS | Latency budget; measured median | Model-size budget; measured | Fits? |
|---|---|---|---|---|---|
| LogisticRegression | Cloud | >= 1 GB; 230.324 MB | <= 100 ms; 0.072450 ms | <= 500 MB; 1.030 KB | Yes |
| LogisticRegression | Edge | 256–1024 MB; 230.324 MB | <= 50 ms; 0.072450 ms | <= 50 MB; 1.030 KB | Yes, at the 256 MB tier with little headroom |
| LogisticRegression | Mobile | 64–256 MB; 230.324 MB | <= 20 ms; 0.072450 ms | <= 10 MB; 1.030 KB | Yes at the upper end of the memory range |
| LogisticRegression | TinyML | <= 256 KB; 230.324 MB | <= 10 ms; 0.072450 ms | <= 100 KB; 1.030 KB | No: process memory exceeds budget |
| RandomForestClassifier | Cloud | >= 1 GB; 231.648 MB | <= 100 ms; 3.881650 ms | <= 500 MB; 284.071 KB | Yes |
| RandomForestClassifier | Edge | 256–1024 MB; 231.648 MB | <= 50 ms; 3.881650 ms | <= 50 MB; 284.071 KB | Yes, at the 256 MB tier with little headroom |
| RandomForestClassifier | Mobile | 64–256 MB; 231.648 MB | <= 20 ms; 3.881650 ms | <= 10 MB; 284.071 KB | Yes at the upper end of the memory range |
| RandomForestClassifier | TinyML | <= 256 KB; 231.648 MB | <= 10 ms; 3.881650 ms | <= 100 KB; 284.071 KB | No: both process memory and model size exceed budget |

LogisticRegression emitted scikit-learn convergence warnings at the mandated
`max_iter=1000` on the unscaled features. The reported score and timing still
use the exact requested baseline; no scaling or estimator changes were made.

## Conclusions

1. LogisticRegression had the higher test accuracy in this split (0.9415 vs.
   0.9357), while RandomForestClassifier trained faster in this environment
   (0.170823 s vs. 0.444112 s).
2. Both models meet the stated Cloud and Edge latency and model-size limits.
   Their measured process memory is about 230–232 MB, so Mobile deployment
   looks possible only on devices with nearly the full 256 MB budget and
   should be rechecked using the intended inference runtime.
3. Neither model meets the TinyML memory budget in this measurement. The
   logistic-regression artifact is small enough for the 100 KB model-size
   limit, but that alone does not make its measured process fit in 256 KB;
   the forest artifact exceeds both TinyML limits.