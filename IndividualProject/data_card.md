# Data Card

## Source

- Dataset: Craigslist Used Cars
- Kaggle: `austinreese/craigslist-carstrucks-data`
- Raw file: `vehicles.csv`
- License and terms: governed by the Kaggle dataset page

## Intended use

The dataset supports a dealership staff tool that proposes a used-car asking
price. Predictions are decision support and should be reviewed by staff.

## Features

The project uses `year`, `manufacturer`, `model`, `condition`, `odometer`,
`fuel`, `transmission`, `drive`, `type`, `paint_color`, and `state`.
Identifiers, URLs, images, descriptions, and free-text fields are excluded.

## Target and exclusions

- Target: `price`
- Valid price range: $1,000 to $150,000
- Valid year range: 1900 through the current year
- Valid odometer range: 0 to 500,000 miles
- Numeric values are coerced and invalid values become missing

## Leakage controls

The target and feature values are filtered before splitting. Imputation,
scaling, and category encoding are fitted inside model pipelines. The test
partition is held out until final model selection and evaluation.

## Known limitations

Listings may contain duplicate vehicles, stale prices, unusual model names,
incorrect odometer readings, and regional selection bias. The dataset does not
guarantee all listings are comparable or recently updated.
