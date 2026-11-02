# Lab 4: Leakage-Free Data Pipeline

## 1. Problem definition

This lab formulates a binary classification task using the Adult dataset
provided by the UCI Machine Learning Repository and accessed through
`fetch_openml(name="adult", version=2, as_frame=True)`. The dataset contains
demographic and employment-related records for individuals. Each example has
14 input features: age, workclass, final weight (`fnlwgt`), education,
education number, marital status, occupation, relationship, race, sex, capital
gain, capital loss, hours worked per week, and native country. The prediction
target is the income category: classify an individual as earning more than
$50,000 per year (`>50K`) or not (`<=50K`). The label is used only as the
supervised-learning outcome and must not be included among the input features.

The objective is to build a reproducible classification workflow that can
prepare mixed numerical and categorical inputs, check that the data conforms
to its expected schema and quality requirements, and evaluate a model on
previously unseen examples. Model performance will be reported using both
accuracy, the proportion of correct predictions, and F1 score, the harmonic
mean of precision and recall for the positive `>50K` class. Reporting both
metrics makes it possible to assess overall correctness as well as the
classifier's performance on the income group of interest.

The central constraint is to prevent data leakage: the test partition must
remain isolated from decisions learned during preprocessing, feature
engineering, model selection, and fitting. Data-dependent transformations
must be fitted using training data only, then applied unchanged to validation
and test data. Splits, preprocessing choices, model settings, and evaluation
must be reproducible from a documented configuration. The resulting workflow
should also make its input checks and processed-data provenance auditable,
including recording a hash of the processed data and documenting the dataset,
preparation, intended use, and limitations in a data card.

## 2. Raw-data ingestion

Run `python src/ingest_validate.py` from the `Lab04` directory. The ingestion
step fetches Adult with `fetch_openml(name="adult", version=2, as_frame=True)`,
retains the returned frame without cleaning, renaming, or dropping records,
and logs its dimensions. The observed raw dataset contains **48,842 rows and
15 columns** (14 predictors and one target). OpenML names the target column
`class`; validation recognizes it as the schema's `income` label without
changing the raw frame.

## 3. Schema and data-quality validation

The script validates the declared 15-column schema: integer fields are
`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, and
`hours-per-week`; the remaining fields, including the income label, are
categorical. The checks include exact column membership and order, compatible
integer/category dtypes, the specified bounds for age (17-90), hours per week
(1-99), capital gain (0-99,999), and capital loss (0-4,356), missingness, and
the permitted income labels (`<=50K` and `>50K`). A literal `?` is accepted in
categorical fields and treated as a missing marker; OpenML parses these
markers as pandas nulls. Missing counts are reported and records are retained.
Failures include a count and up to 20 example rows or values; the full report
is written to [`results/validation_report.json`](./results/validation_report.json).

On the fetched data, all schema, type, range, and income-label checks passed
with **zero violations**. The frame contains 2,799 missing values in
`workclass`, 2,809 in `occupation`, and 857 in `native-country`. No literal
`?` values remained after OpenML parsing. **No rows were dropped.**

## 4. Train/test cleaning and transformation

The same command performs a reproducible stratified 70/30 split (seed 42)
before cleaning. It removes exact duplicate records from the training
partition only; the test partition is not deduplicated or otherwise modified.
On this run, 24 duplicate training rows were removed (34,189 to 34,165);
all 14,653 test rows were retained. Numeric imputation medians, categorical
imputation modes, and numeric clipping bounds (training-set 1st/99th
percentiles) are fitted using training data only. The fitted transformer is
saved to [`results/preprocessor.joblib`](./results/preprocessor.joblib), then
applied consistently to the held-out test features.

## 5. Derived features and baseline

The feature-engineered model adds `capital_net` (`capital-gain` minus
`capital-loss`) and `hours_per_age` (`hours-per-week` divided by `age - 16`).
Both are derived within the transformer before numeric imputation, and their
clipping/imputation parameters are fitted on training data. LogisticRegression
is evaluated with accuracy and positive-class F1, both before and after feature
engineering:

| Model inputs | Accuracy | F1 (`>50K`) |
|---|---:|---:|
| Original 14 features | 0.8540 | 0.6601 |
| Original plus two derived features | 0.8578 | 0.6696 |

These are results from the fixed split and configuration, not a claim of
generalization beyond this dataset.

## 6. Label-quality audit

The full dataset was grouped by the 14 predictors to detect records with
identical features but conflicting income labels. The audit found **5
conflicting feature groups (10 records in those groups)**. A reproducible
random sample of 50 rows (seed 42) was checked against the Adult categorical
domains and documented numeric bounds; **0 rows were flagged**. The sample
values and per-row checks are
available in [`results/pipeline_results.json`](./results/pipeline_results.json).
This rule-based review can identify impossible values, but does not independently
verify that census labels are factually correct.

## 7. Hash, configuration, and data card

The SHA256 of the transformed train/test feature matrix, row ordering, split
membership, and labels is recorded as
`0256b280962fdae8e88fcfc8978a9b3fd01775e1a6cce1ba6e0942266ea1e166` in
[`results/pipeline_config.yaml`](./results/pipeline_config.yaml) alongside the
run settings. The metrics, sample audit, and split counts are in
[`results/pipeline_results.json`](./results/pipeline_results.json). Dataset
source, license, size, intended use, known biases, and privacy considerations
are documented in [`data_card.md`](./data_card.md).

Run the complete pipeline from the `Lab04` directory with:

```powershell
python src\ingest_validate.py
```