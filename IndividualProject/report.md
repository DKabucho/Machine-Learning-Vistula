# Individual Project: Used-Car Price Estimation

**Student:** Daniel Kabucho Karanja

**Student ID:** 70470

## Problem

Build a used-car price-estimation tool for a car dealership. Given listing
details such as year, manufacturer, model, condition, mileage, fuel type,
transmission, and location, predict the asking price in US dollars. The service
will support dealership staff when they prepare a vehicle listing.

## Dataset

Use the Kaggle dataset
[Craigslist Used Cars](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data),
identified as `austinreese/craigslist-carstrucks-data`. The raw file is
`vehicles.csv`, with approximately 426,000 listings. Because listing quality
varies, the final work will document excluded records and data transformations.

Candidate predictors are `year`, `manufacturer`, `model`, `condition`,
`odometer`, `fuel`, `transmission`, `drive`, `type`, `paint_color`, and
`state`. Identifier, URL, image, and free-text fields are excluded from the
baseline. Price is the target.

## Success metric

The primary success metric is mean absolute error (MAE) on a held-out test
partition. Report median absolute error, RMSE, and R-squared as secondary
metrics. The proposed target is **MAE no greater than $3,000**.

## Deployment constraints

- Prediction latency no greater than 50 ms per vehicle
- Serialized model and preprocessing pipeline no greater than 5 MB
- CPU-only first deployment
- Structured web-form input rather than image input

These limits include preprocessing and inference in the final deployment
measurement, not just model prediction time.

## Implemented comparison and optimization

The project compares a regularized Ridge baseline with a compact histogram-
gradient-boosting model. Ridge uses median imputation, scaling, and compact
one-hot encoding. Histogram gradient boosting uses median imputation and
ordinal encoding. Both preprocessing and model stages are contained in a
single serialized scikit-learn pipeline.

The expanded design also includes a compact tabular MLP, following the neural-
network, automatic-differentiation, profiling, and YAML/MLflow concepts from
Labs 2, 5, and 6. The MLP is appropriate for structured attributes, unlike a
CNN or patch attention model, whose main benefit would come from image or text
inputs not included in the baseline dataset.

The selected model is the candidate with the lowest validation MAE. It is then
retrained on the combined training and validation partitions and evaluated once
on the held-out test partition. If it exceeds the 5 MB or 50 ms limit, the
project will evaluate quantization, pruning, categorical vocabulary reduction,
feature reduction, or smaller MLP/boosting complexity and repeat the same
measurements. Neural distillation is a fallback when a larger MLP improves
validation MAE enough to justify training a compact student.

No approach is considered successful until MAE, secondary metrics, size, and
latency have been measured with the held-out test set and deployment-like CPU
environment.

## Experimental plan

1. Download `vehicles.csv` and record the file hash and row count.
2. Use seed 42 for splitting; clean training and test data consistently.
3. Remove implausible or missing targets and duplicate identifiers.
4. Fit missing-value and category transformations using training data only.
5. Compare a regularized linear baseline and a compact boosted baseline.
6. Select the model with the lowest validation MAE that remains feasible.
7. Apply a compact-size optimization if required.
8. Report final test MAE, median AE, RMSE, R-squared, size, and p95 latency.

## Reproducibility

`config.yaml` records the dataset, split seed, target, filtering limits, and
deployment targets. Raw data and generated results are not committed to Git.
The final report must include Kaggle terms, dataset hash, row-count filters,
and the exact preprocessing pipeline.

Download the dataset with Kaggle authentication configured:

```powershell
kaggle datasets download -d austinreese/craigslist-carstrucks-data -p IndividualProject\data --unzip
```

The repository's baseline script expects `IndividualProject\data\vehicles.csv`.
To train it after downloading the Kaggle file, run:

```powershell
& '.\.venv\Scripts\python.exe' '.\IndividualProject\src\run_baseline.py'
```

## Implemented pipeline

The baseline now follows a leak-safe, reproducible workflow:

1. Record the raw-row count and SHA-256 hash of `vehicles.csv`.
2. Remove invalid prices, years, and odometer values.
3. Create deterministic 60/20/20 train, validation, and test partitions using
   seed 42.
4. Fit missing-value and encoding transformations only inside each model
   pipeline.
5. Compare regularized Ridge regression against histogram gradient boosting.
6. Select the model with lower validation MAE, retrain it on train plus
   validation data, and evaluate the test partition once.
7. Measure serialized pipeline size and 200 one-row CPU requests, reporting
   the p95 latency.
8. Save `baseline-metrics.json` and `model-comparison.csv` for reproducible
   reporting.

This structure applies the environment and cost measurement approach from Lab
01, leakage-safe preprocessing and validation principles from Lab 04,
reproducible configuration and benchmarking from Lab 06, and a deterministic
train/validation/test workflow to the tabular project.

Run the focused tests with:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest '.\IndividualProject\tests'
```
