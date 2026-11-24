# Model Card

## Intended use

The model proposes used-car asking prices for dealership staff. It is a
decision-support tool, not an automated valuation or an appraisal.

## Users and environment

- Primary users: dealership listing staff
- Intended environment: CPU-only structured web-form service
- Input: year, manufacturer, model, condition, odometer, fuel, transmission,
  drive, vehicle type, paint color, and state
- Output: estimated asking price in US dollars

## Candidate models

- Regularized Ridge regression
- Histogram gradient boosting
- Compact tabular MLP

The Ridge and boosting models are measured as classical baselines. The MLP
uses two hidden layers and is compared for validation MAE, parameter count,
latency, memory, and serialized size.

## Metrics

Primary:
- Mean absolute error, with a $3,000 target

Secondary:
- Median absolute error
- RMSE
- R-squared
- P95 single-request latency
- Peak memory
- Serialized pipeline size

## Data

See `data_card.md`. Training and validation transformations are fitted only
from training data. The test partition is reserved for one final evaluation.

## Limitations

The model may reflect regional, temporal, and marketplace bias in listing data.
It should not be used for financing decisions, insurance claims, fraud
decisions, or appraisal without independent review.

## Deployment

The target is at most 50 ms per request and at most 5 MB for the serialized
model plus preprocessing pipeline. Quantization, pruning, distillation,
feature reduction, or a smaller architecture may be applied only if measured
and compared against the deployment targets.
