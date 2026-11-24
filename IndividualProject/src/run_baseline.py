from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
RANDOM_STATE = 42
FEATURES = [
    "year",
    "manufacturer",
    "model",
    "condition",
    "odometer",
    "fuel",
    "transmission",
    "drive",
    "type",
    "paint_color",
    "state",
]
NUMERIC_FEATURES = ["year", "odometer"]
CATEGORICAL_FEATURES = [name for name in FEATURES if name not in NUMERIC_FEATURES]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    config["_project_dir"] = str(path.resolve().parent)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config[key])
    return value if value.is_absolute() else Path(config["_project_dir"]) / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    target = config["target_column"]
    raw_rows = int(len(frame))
    frame = frame[list(FEATURES) + [target]].copy()
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_price = frame[target].between(
        int(config["minimum_price"]),
        int(config["maximum_price"]),
    )
    valid_year = frame["year"].between(
        int(config["minimum_year"]),
        int(pd.Timestamp.now().year),
    )
    valid_odometer = frame["odometer"].between(
        0,
        int(config["maximum_odometer"]),
    )
    frame = frame[valid_price & valid_year & valid_odometer]

    stats = {
        "raw_rows": raw_rows,
        "excluded_rows": raw_rows - int(len(frame)),
        "clean_rows": int(len(frame)),
    }
    return frame, stats


def make_models(config: dict[str, Any]) -> dict[str, Pipeline]:
    ridge_numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ridge_categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=100,
                    sparse_output=True,
                ),
            ),
        ]
    )
    ridge_preprocessing = ColumnTransformer(
        [
            ("numeric", ridge_numeric, NUMERIC_FEATURES),
            ("categorical", ridge_categorical, CATEGORICAL_FEATURES),
        ]
    )

    ridge_config = config["models"]["ridge"]
    ridge = Pipeline(
        [
            ("preprocessing", ridge_preprocessing),
            ("regressor", Ridge(alpha=float(ridge_config["alpha"]))),
        ]
    )

    ordinal_numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    ordinal_categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    boosting_preprocessing = ColumnTransformer(
        [
            ("numeric", ordinal_numeric, NUMERIC_FEATURES),
            ("categorical", ordinal_categorical, CATEGORICAL_FEATURES),
        ]
    )

    boosting_config = config["models"]["histogram_gradient_boosting"]
    boosting = Pipeline(
        [
            ("preprocessing", boosting_preprocessing),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=float(boosting_config["learning_rate"]),
                    max_iter=int(boosting_config["max_iter"]),
                    max_depth=int(boosting_config["max_depth"]),
                    l2_regularization=float(
                        boosting_config["l2_regularization"]
                    ),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {"ridge": ridge, "histogram_gradient_boosting": boosting}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a reproducible used-car price baseline."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--csv",
        type=Path,
        help="Override the vehicles.csv path.",
    )
    args = parser.parse_args()
    config = load_config(args.config.resolve())

    csv_path = args.csv or resolve_path(config, "dataset_file")
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            f"Download the Kaggle dataset {config['dataset']['kaggle_slug']} "
            "and place vehicles.csv at this path."
        )

    frame = pd.read_csv(csv_path)
    required = set(FEATURES + [config["target_column"]])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame, data_stats = clean_frame(frame, config)
    if len(frame) < 100:
        raise ValueError(
            f"Only {len(frame)} rows remain after cleaning; at least 100 are required."
        )

    train_validation, test_frame = train_test_split(
        frame,
        test_size=float(config["test_size"]),
        random_state=RANDOM_STATE,
    )
    validation_fraction = float(config["validation_size"]) / (
        1.0 - float(config["test_size"])
    )
    train_frame, validation_frame = train_test_split(
        train_validation,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
    )

    target = config["target_column"]
    x_train = train_frame[FEATURES]
    y_train = train_frame[target]
    x_validation = validation_frame[FEATURES]
    y_validation = validation_frame[target]
    x_test = test_frame[FEATURES]
    y_test = test_frame[target]

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    output_dir = resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = []

    for name, model in make_models(config).items():
        model.fit(x_train, y_train)
        validation_predictions = model.predict(x_validation)
        comparison.append(
            {
                "model": name,
                "validation_mae_usd": mean_absolute_error(
                    y_validation, validation_predictions
                ),
            }
        )

    best_name = min(
        comparison,
        key=lambda row: row["validation_mae_usd"],
    )["model"]

    x_train_final = train_validation[FEATURES]
    y_train_final = train_validation[target]
    final_model = make_models(config)[best_name]
    final_model.fit(x_train_final, y_train_final)

    predictions = final_model.predict(x_test)
    errors = np.abs(y_test.to_numpy() - predictions)
    mae = mean_absolute_error(y_test, predictions)
    median_ae = float(np.median(errors))
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r_squared = r2_score(y_test, predictions)

    latencies = []
    sample = x_test.iloc[:1]
    for _ in range(20):
        final_model.predict(sample)
    for _ in range(200):
        started = time.perf_counter_ns()
        final_model.predict(sample)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    latencies.sort()
    p95_index = min(len(latencies) - 1, int(np.ceil(0.95 * len(latencies))) - 1)
    p95_latency = latencies[p95_index]

    model_path = output_dir / "used-car-selected.joblib"
    joblib.dump(final_model, model_path)
    size_mb = model_path.stat().st_size / (1024**2)

    result = {
        "dataset": config["dataset"]["name"],
        "dataset_kaggle_slug": config["dataset"]["kaggle_slug"],
        "dataset_file": str(csv_path),
        "dataset_sha256": sha256(csv_path),
        "random_seed": RANDOM_STATE,
        "raw_rows": data_stats["raw_rows"],
        "excluded_rows": data_stats["excluded_rows"],
        "clean_rows": data_stats["clean_rows"],
        "model_selection_train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "final_train_rows": int(len(train_validation)),
        "test_rows": int(len(test_frame)),
        "selected_model": best_name,
        "validation_comparison": comparison,
        "mae_usd": mae,
        "median_absolute_error_usd": median_ae,
        "rmse_usd": rmse,
        "r_squared": r_squared,
        "single_request_p95_latency_ms": p95_latency,
        "serialized_size_mb": size_mb,
        "mae_target_pass": mae <= float(config["mae_target_usd"]),
        "latency_target_pass": p95_latency
        <= float(config["latency_target_ms"]),
        "size_target_pass": size_mb <= float(config["size_target_mb"]),
    }

    with (output_dir / "baseline-metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)

    with (output_dir / "model-comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "validation_mae_usd"],
        )
        writer.writeheader()
        writer.writerows(comparison)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
