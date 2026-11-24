from __future__ import annotations

import argparse
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
from sklearn.preprocessing import OrdinalEncoder


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

    raw_rows = int(len(frame))
    frame = frame[list(FEATURES) + [config["target_column"]]].copy()
    frame = frame[
        frame[config["target_column"]].between(
            int(config["minimum_price"]),
            int(config["maximum_price"]),
        )
    ]
    frame = frame.dropna(subset=[config["target_column"]])
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    train_frame, test_frame = train_test_split(
        frame,
        test_size=float(config["test_size"]),
        random_state=RANDOM_STATE,
    )
    x_train = train_frame[FEATURES]
    y_train = train_frame[config["target_column"]]
    x_test = test_frame[FEATURES]
    y_test = test_frame[config["target_column"]]

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
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
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.08,
                    max_iter=200,
                    max_depth=8,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    errors = np.abs(y_test.to_numpy() - predictions)
    mae = mean_absolute_error(y_test, predictions)
    median_ae = float(np.median(errors))
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r_squared = r2_score(y_test, predictions)

    latencies = []
    sample = x_test.iloc[:1]
    for _ in range(20):
        model.predict(sample)
    for _ in range(200):
        started = time.perf_counter_ns()
        model.predict(sample)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    latencies.sort()
    p95_index = min(len(latencies) - 1, int(np.ceil(0.95 * len(latencies))) - 1)

    output_dir = resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "used-car-baseline.joblib"
    joblib.dump(model, model_path)
    size_mb = model_path.stat().st_size / (1024**2)

    result = {
        "dataset": config["dataset"]["name"],
        "dataset_kaggle_slug": config["dataset"]["kaggle_slug"],
        "dataset_file": str(csv_path),
        "dataset_sha256": sha256(csv_path),
        "raw_rows": raw_rows,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "mae_usd": mae,
        "median_absolute_error_usd": median_ae,
        "rmse_usd": rmse,
        "r_squared": r_squared,
        "single_request_p95_latency_ms": latencies[p95_index],
        "serialized_size_mb": size_mb,
        "mae_target_pass": mae <= float(config["mae_target_usd"]),
        "latency_target_pass": latencies[p95_index]
        <= float(config["latency_target_ms"]),
        "size_target_pass": size_mb <= float(config["size_target_mb"]),
    }

    with (output_dir / "baseline-metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
