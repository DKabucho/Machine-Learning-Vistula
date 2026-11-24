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

## Planned optimization

Begin with a compact histogram-gradient-boosting or linear baseline and compare
its test error, size, and latency with a regularized linear model. If the
stronger model does not meet the 5 MB or 50 ms limits, use a smaller feature
set or smaller boosted model. Distillation or structured pruning may be added
only if a larger neural baseline is evaluated and is demonstrably useful.

No approach is considered successful until test accuracy, size, and latency
have been measured with the held-out test set and deployment-like environment.

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
