# Adult Dataset Data Card

## Dataset and source

- **Name:** Adult (Census Income)
- **Source:** UCI Machine Learning Repository, loaded through OpenML with
  `fetch_openml(name="adult", version=2, as_frame=True)`.
- **Original data:** 1994 US Census records prepared by Barry Becker.
- **License:** CC BY 4.0 as listed by the
  [UCI Adult dataset page](https://archive.ics.uci.edu/dataset/2/adult).
- **Size:** 48,842 rows, 14 predictor fields, and one income label.
- **Label:** income above $50,000 (`>50K`) versus at most $50,000 (`<=50K`).
- **Missingness:** OpenML represents source `?` markers as nulls for workclass,
  occupation, and native-country. The pipeline imputes these from training
  data; it does not delete test rows.

## Processing and evaluation

The pipeline uses a reproducible 70/30 stratified split (seed 42) before
cleaning. Exact duplicates are removed from the training partition only.
Numeric medians, categorical modes, and 1st/99th percentile clipping bounds
are fitted on the remaining training rows, then reused for test transformation.
The saved `results/preprocessor.joblib` includes imputation, clipping,
categorical encoding, numeric scaling, and (for the primary transformer)
feature engineering. Accuracy and positive-class F1 are reported for a
LogisticRegression baseline and for the model with `capital_net` and
`hours_per_age`.

The processed-data SHA256, run configuration, split counts, metrics, and
label-quality audit are recorded in `results/pipeline_config.yaml` and
`results/pipeline_results.json`. The SHA256 is over the transformed train and
test feature matrix, its row/split ordering, and labels.

## Intended use

This dataset and pipeline are for coursework in data validation, reproducible
preprocessing, feature engineering, and binary-classification evaluation.
They are not suitable for making decisions about an individual person's
employment, credit, benefits, or other access to services.

## Known limitations and biases

The records are historical US census data and do not represent current
economic conditions or populations outside the dataset's collection context.
Income is affected by structural and demographic factors. Race, sex, age,
marital status, and country of origin are present, and other features can act
as proxies for protected or sensitive attributes. Class imbalance and
historical inequities can make aggregate scores obscure disparate error rates.
This lab reports accuracy and F1, not a complete fairness assessment.

## Privacy and ethical considerations

Although the source is publicly available, census-derived demographic and
income information is sensitive. Do not attempt to identify individuals,
combine records with external identifying sources, or deploy predictions for
high-impact decisions. Treat intermediate data and model outputs as sensitive,
and share only the minimum information required for coursework.

## Audit notes

The run's 50-row reproducible sample is checked against the Adult field
domains and documented numeric bounds. The duplicate-conflict audit groups
records by all predictors and counts groups whose income labels disagree.
Detailed sampled values, findings, and conflict counts are in
`results/pipeline_results.json`.
