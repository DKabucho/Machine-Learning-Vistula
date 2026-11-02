"""Ingest the raw Adult dataset and report schema and data-quality checks."""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))
from src.preprocessing import AdultPreprocessor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

EXPECTED_TYPES = {
    "age": "int",
    "workclass": "category",
    "fnlwgt": "int",
    "education": "category",
    "education-num": "int",
    "marital-status": "category",
    "occupation": "category",
    "relationship": "category",
    "race": "category",
    "sex": "category",
    "capital-gain": "int",
    "capital-loss": "int",
    "hours-per-week": "int",
    "native-country": "category",
    "income": "category",
}
RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}
VALID_INCOME_LABELS = {"<=50K", ">50K"}
MISSING_MARKER = "?"
RANDOM_STATE = 42
TEST_SIZE = 0.30
SAMPLE_SIZE = 50
NUMERIC_FEATURES = (
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
)
CATEGORICAL_FEATURES = tuple(
    column for column, kind in EXPECTED_TYPES.items()
    if kind == "category" and column != "income"
)
DERIVED_FEATURES = ("capital_net", "hours_per_age")
FEATURE_DOMAINS = {
    "workclass": {
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked",
    },
    "education": {
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
        "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool",
    },
    "marital-status": {
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse",
    },
    "occupation": {
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces",
    },
    "relationship": {
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
        "Unmarried",
    },
    "race": {
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black",
    },
    "sex": {"Female", "Male"},
    "native-country": {
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
        "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
        "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
        "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
        "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
        "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland",
        "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru",
        "Hong", "Holand-Netherlands",
    },
}


def _is_category_dtype(dtype: Any) -> bool:
    """Accept pandas categorical columns and string-like OpenML columns."""
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or is_object_dtype(dtype)
        or is_string_dtype(dtype)
    )


def _check(name: str, violations: list[Any], **details: Any) -> dict[str, Any]:
    return {
        "check": name,
        "status": "PASS" if not violations else "FAIL",
        "violation_count": len(violations),
        "sample_violations": [str(value) for value in violations[:20]],
        **details,
    }


def validate_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate the fetched frame without changing it or removing any rows."""
    checks: list[dict[str, Any]] = []
    expected_columns = list(EXPECTED_TYPES)
    source_label_column = (
        "income" if "income" in frame else "class" if "class" in frame else "income"
    )
    source_columns = [
        source_label_column if column == "income" else column
        for column in expected_columns
    ]
    missing_columns = [column for column in source_columns if column not in frame]
    extra_columns = [column for column in frame if column not in source_columns]
    wrong_order = (
        not missing_columns
        and not extra_columns
        and list(frame.columns) != source_columns
    )
    column_violations = missing_columns + extra_columns
    if wrong_order:
        column_violations.append("column order differs from declared schema")
    checks.append(
        _check(
            "schema columns",
            column_violations,
            expected_column_count=len(expected_columns),
            observed_column_count=len(frame.columns),
            missing_columns=missing_columns,
            extra_columns=extra_columns,
            wrong_order=wrong_order,
            source_label_column=source_label_column,
        )
    )

    for column, expected_type in EXPECTED_TYPES.items():
        source_column = source_label_column if column == "income" else column
        if source_column not in frame:
            checks.append(
                _check(
                    f"type: {column}",
                    [f"column {column!r} is absent"],
                    expected_type=expected_type,
                )
            )
            continue
        dtype = frame[source_column].dtype
        matches = (
            is_integer_dtype(dtype)
            if expected_type == "int"
            else _is_category_dtype(dtype)
        )
        checks.append(
            _check(
                f"type: {column}",
                [] if matches else [f"observed dtype {dtype}"],
                expected_type=expected_type,
                observed_dtype=str(dtype),
            )
        )

    for column, (minimum, maximum) in RANGES.items():
        if column not in frame:
            violations = [f"column {column!r} is absent"]
        else:
            values = frame[column]
            out_of_range = values.notna() & ~values.between(minimum, maximum)
            violations = [
                f"row {index}: {frame.at[index, column]!r}"
                for index in frame.index[out_of_range][:20]
            ]
            # Keep the full count even though only a bounded sample is logged.
            violation_count = int(out_of_range.sum())
            checks.append(
                {
                    "check": f"range: {column}",
                    "status": "PASS" if violation_count == 0 else "FAIL",
                    "violation_count": violation_count,
                    "sample_violations": violations,
                    "minimum": minimum,
                    "maximum": maximum,
                }
            )
            continue
        checks.append(
            _check(
                f"range: {column}",
                violations,
                minimum=minimum,
                maximum=maximum,
            )
        )

    null_columns = {
        column: int(count)
        for column, count in frame.isna().sum().items()
        if count
    }
    checks.append(
        _check(
            "actual null values",
            [],
            nulls_by_column=null_columns,
            policy=(
                "Missing values are retained and reported; OpenML may parse the "
                "source marker '?' as pandas NA."
            ),
        )
    )

    marker_counts = {
        column: int((frame[column] == MISSING_MARKER).sum())
        for column, kind in EXPECTED_TYPES.items()
        if kind == "category" and column in frame
    }
    checks.append(
        _check(
            "missing marker '?'",
            [],
            marker_counts_by_column={
                column: count for column, count in marker_counts.items() if count
            },
            policy="'?' is counted as missing and accepted in categorical columns.",
        )
    )

    if source_label_column not in frame:
        checks.append(
            _check(
                "income label quality",
                [f"{source_label_column} column is absent"],
                allowed_labels=sorted(VALID_INCOME_LABELS),
                observed_labels=[],
                source_column=source_label_column,
            )
        )
    else:
        labels = frame[source_label_column]
        invalid = labels.isna() | ~labels.isin(VALID_INCOME_LABELS)
        sample_violations = [
            f"row {index}: {frame.at[index, source_label_column]!r}"
            for index in frame.index[invalid][:20]
        ]
        observed_labels = sorted(
            str(label) for label in labels.dropna().unique()
        )
        checks.append(
            {
                "check": "income label quality",
                "status": "PASS" if int(invalid.sum()) == 0 else "FAIL",
                "violation_count": int(invalid.sum()),
                "sample_violations": sample_violations,
                "allowed_labels": sorted(VALID_INCOME_LABELS),
                "observed_labels": observed_labels,
                "source_column": source_label_column,
            }
        )

    return {
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "rows_dropped": 0,
        "checks": checks,
    }


def _source_label_column(frame: pd.DataFrame) -> str:
    if "income" in frame:
        return "income"
    if "class" in frame:
        return "class"
    raise ValueError("Adult data is missing both 'income' and OpenML's 'class' label.")


def _quality_audit(frame: pd.DataFrame, label_column: str) -> dict[str, Any]:
    features = [column for column in frame.columns if column != label_column]
    duplicate_rows = frame.duplicated(subset=features, keep=False)
    conflict_summary = (
        frame.loc[duplicate_rows]
        .groupby(features, dropna=False, observed=True, sort=False)[label_column]
        .agg(
            label_count=lambda values: values.nunique(dropna=False),
            row_count="size",
        )
    )
    conflicting_mask = conflict_summary["label_count"] > 1
    conflicting_group_count = int(conflicting_mask.sum())
    conflicting_row_count = int(
        conflict_summary.loc[conflicting_mask, "row_count"].sum()
    )

    sample = frame.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    reviewed_rows: list[dict[str, Any]] = []
    numeric_ranges = {
        "age": (17, 90),
        "fnlwgt": (1, None),
        "education-num": (1, 16),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
        "hours-per-week": (1, 99),
    }
    for row_index, row in sample.iterrows():
        violations: list[str] = []
        for column, (minimum, maximum) in numeric_ranges.items():
            value = row[column]
            if pd.isna(value):
                violations.append(f"{column} is missing")
            elif value < minimum or (maximum is not None and value > maximum):
                violations.append(
                    f"{column}={value!r} outside [{minimum}, {maximum}]"
                )
        for column, allowed in FEATURE_DOMAINS.items():
            value = row[column]
            if pd.notna(value) and value != MISSING_MARKER and value not in allowed:
                violations.append(f"{column}={value!r} is not in the Adult dictionary")
        label = row[label_column]
        if pd.isna(label) or label not in VALID_INCOME_LABELS:
            violations.append(f"{label_column}={label!r} is not a valid income label")
        reviewed_rows.append(
            {
                "row_index": int(row_index),
                "values": {
                    column: ("?" if pd.isna(value) else value)
                    for column, value in row.items()
                },
                "violations": violations,
                "review_result": "FLAG" if violations else "PASS",
            }
        )

    return {
        "duplicate_feature_label_conflicts": {
            "conflicting_feature_groups": conflicting_group_count,
            "rows_in_conflicting_groups": conflicting_row_count,
        },
        "sample_review": {
            "sample_size": len(reviewed_rows),
            "random_state": RANDOM_STATE,
            "method": (
                "Reproducible 50-row review against the Adult field domains and "
                "numeric bounds encoded in this script; missing categorical "
                "values are allowed."
            ),
            "rows_flagged": sum(bool(item["violations"]) for item in reviewed_rows),
            "rows": reviewed_rows,
        },
    }


def _fit_and_score(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    engineer_features: bool,
) -> tuple[AdultPreprocessor, dict[str, float]]:
    numeric_features = NUMERIC_FEATURES + (
        DERIVED_FEATURES if engineer_features else ()
    )
    preprocessor = AdultPreprocessor(
        numeric_features=numeric_features,
        categorical_features=CATEGORICAL_FEATURES,
        engineer_features=engineer_features,
    )
    train_matrix = preprocessor.fit_transform(X_train)
    test_matrix = preprocessor.transform(X_test)
    classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    classifier.fit(train_matrix, y_train)
    predictions = classifier.predict(test_matrix)
    return preprocessor, {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, pos_label=1)),
    }


def _processed_data_hash(
    preprocessor: AdultPreprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> str:
    from scipy import sparse

    train_matrix = preprocessor.transform(X_train).tocsr()
    test_matrix = preprocessor.transform(X_test).tocsr()
    matrix = sparse.vstack([train_matrix, test_matrix], format="csr")
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.data.tobytes())
    digest.update(matrix.indices.tobytes())
    digest.update(matrix.indptr.tobytes())
    digest.update(
        json.dumps(
            {
                "train_indices": [int(index) for index in X_train.index],
                "test_indices": [int(index) for index in X_test.index],
                "train_labels": [int(label) for label in y_train],
                "test_labels": [int(label) for label in y_test],
                "features": preprocessor.get_feature_names_out().tolist(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _write_config(path: Path, data_hash: str, audit: dict[str, Any]) -> None:
    content = f"""dataset:
  name: Adult
  loader: fetch_openml
  version: 2
  expected_rows: 48842
split:
  test_size: {TEST_SIZE}
  stratified: true
  random_state: {RANDOM_STATE}
cleaning:
  drop_exact_duplicates_from_train_only: true
  categorical_imputation: mode
  numeric_imputation: median
  missing_marker: "?"
  numeric_clipping_percentiles: [1, 99]
features:
  engineered: [capital_net, hours_per_age]
model:
  name: LogisticRegression
  max_iter: 1000
label_audit:
  sample_size: {SAMPLE_SIZE}
  random_state: {RANDOM_STATE}
  duplicate_conflicting_feature_groups: {audit["duplicate_feature_label_conflicts"]["conflicting_feature_groups"]}
processed_data:
  sha256: {data_hash}
  hash_includes: transformed_train_and_test_features_labels_and_row_indices
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    raw_frame = dataset.frame

    # Ingestion only observes the fetched frame; it does not clean or transform it.
    LOGGER.info(
        "Ingested raw Adult data: %d rows, %d columns",
        len(raw_frame),
        len(raw_frame.columns),
    )
    report = validate_frame(raw_frame)
    for result in report["checks"]:
        LOGGER.info(
            "%s: %s (%d violation(s))",
            result["check"],
            result["status"],
            result["violation_count"],
        )
        if result["sample_violations"]:
            LOGGER.warning(
                "%s sample violations: %s",
                result["check"],
                "; ".join(result["sample_violations"]),
            )

    output_dir = LAB_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "validation_report.json"
    validation_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    label_column = _source_label_column(raw_frame)
    audit = _quality_audit(raw_frame, label_column)

    labels = raw_frame[label_column].map({"<=50K": 0, ">50K": 1})
    if labels.isna().any():
        raise ValueError("Income label check failed; refusing to train on invalid labels.")
    train_indices, test_indices = train_test_split(
        np.arange(len(raw_frame)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    train_raw = raw_frame.iloc[train_indices].copy()
    test_raw = raw_frame.iloc[test_indices].copy()
    train_raw = train_raw.drop_duplicates(keep="first")
    X_train = train_raw.drop(columns=[label_column])
    y_train = train_raw[label_column].map({"<=50K": 0, ">50K": 1})
    X_test = test_raw.drop(columns=[label_column])
    y_test = test_raw[label_column].map({"<=50K": 0, ">50K": 1})

    _, baseline_metrics = _fit_and_score(
        X_train, y_train, X_test, y_test, engineer_features=False
    )
    fitted_preprocessor, engineered_metrics = _fit_and_score(
        X_train, y_train, X_test, y_test, engineer_features=True
    )
    data_hash = _processed_data_hash(
        fitted_preprocessor, X_train, y_train, X_test, y_test
    )
    joblib.dump(fitted_preprocessor, output_dir / "preprocessor.joblib")

    results = {
        "dataset": {"rows": len(raw_frame), "columns": len(raw_frame.columns)},
        "split": {
            "train_rows_before_deduplication": len(train_indices),
            "train_rows_after_deduplication": len(train_raw),
            "train_exact_duplicates_removed": len(train_indices) - len(train_raw),
            "test_rows": len(test_raw),
            "test_modified_or_deduplicated": False,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "test_size": TEST_SIZE,
        },
        "metrics": {
            "without_engineered_features": baseline_metrics,
            "with_engineered_features": engineered_metrics,
        },
        "label_quality_audit": audit,
        "processed_data_sha256": data_hash,
        "rows_dropped_from_test": 0,
    }
    (output_dir / "pipeline_results.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    _write_config(output_dir / "pipeline_config.yaml", data_hash, audit)
    LOGGER.info("Validation report saved to %s", validation_path)
    LOGGER.info("Train rows after exact-deduplication: %d", len(train_raw))
    LOGGER.info("Test rows retained untouched: %d", len(test_raw))
    LOGGER.info(
        "LogisticRegression baseline: accuracy=%.4f, F1=%.4f",
        baseline_metrics["accuracy"],
        baseline_metrics["f1"],
    )
    LOGGER.info(
        "With derived features: accuracy=%.4f, F1=%.4f",
        engineered_metrics["accuracy"],
        engineered_metrics["f1"],
    )
    LOGGER.info(
        "Duplicate feature groups with conflicting labels: %d",
        audit["duplicate_feature_label_conflicts"]["conflicting_feature_groups"],
    )
    LOGGER.info(
        "50-row dictionary audit flagged: %d",
        audit["sample_review"]["rows_flagged"],
    )
    LOGGER.info("Processed-data SHA256: %s", data_hash)


if __name__ == "__main__":
    main()
