from __future__ import annotations

import csv
import random
import statistics
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import joblib
import numpy as np
import psutil
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPOSITORY_ROOT / "Lab01" / "results"
WARMUP_RUNS = 1
TRAINING_RUNS = 5
INFERENCE_RUNS = 100
RSS_SAMPLE_INTERVAL_SECONDS = 0.001

Model = TypeVar("Model", LogisticRegression, RandomForestClassifier)
Result = TypeVar("Result")


def seed_everything() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def measure_operation(operation: Callable[[], Result]) -> tuple[Result, float, int]:
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = [baseline_rss]
    stop_sampling = threading.Event()

    def sample_rss() -> None:
        while not stop_sampling.is_set():
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            stop_sampling.wait(RSS_SAMPLE_INTERVAL_SECONDS)

    sampler = threading.Thread(target=sample_rss)
    sampler.start()
    started = time.perf_counter()
    try:
        result = operation()
    finally:
        elapsed_seconds = time.perf_counter() - started
        stop_sampling.set()
        sampler.join()
    peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    return result, elapsed_seconds, peak_rss[0]


def model_factories() -> dict[str, Callable[[], Model]]:
    return {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
    }


def main() -> None:
    seed_everything()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    features, labels = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=42,
    )

    models: dict[str, Model] = {}
    accuracy_rows: list[dict[str, str]] = []
    measurement_rows: list[dict[str, str | int]] = []

    for name, make_model in model_factories().items():
        for _ in range(WARMUP_RUNS):
            warmup_model = make_model()
            warmup_model.fit(x_train, y_train)

        training_times: list[float] = []
        training_peaks: list[int] = []
        fitted_model: Model | None = None
        for _ in range(TRAINING_RUNS):
            model = make_model()
            fitted, elapsed, peak_rss = measure_operation(
                lambda model=model: model.fit(x_train, y_train)
            )
            if fitted is not model:
                raise RuntimeError(f"Unexpected fitted model returned for {name}")
            fitted_model = model
            training_times.append(elapsed)
            training_peaks.append(peak_rss)

        if fitted_model is None:
            raise RuntimeError(f"No fitted model available for {name}")
        models[name] = fitted_model
        accuracy = fitted_model.score(x_test, y_test)
        accuracy_rows.append({"model": name, "test_accuracy": f"{accuracy:.4f}"})

        sample = x_test[:1]
        fitted_model.predict(sample)
        inference_times: list[float] = []

        def run_inference_samples() -> np.ndarray:
            for _ in range(INFERENCE_RUNS):
                started = time.perf_counter()
                fitted_model.predict(sample)
                inference_times.append(time.perf_counter() - started)
            return np.asarray(inference_times)

        _, _, inference_peak_rss = measure_operation(run_inference_samples)

        individual_model_path = RESULTS_DIR / f"{name.lower()}_model.joblib"
        joblib.dump(fitted_model, individual_model_path)
        model_bytes = individual_model_path.stat().st_size

        measurement_rows.append(
            {
                "model": name,
                "training_runs": TRAINING_RUNS,
                "training_median_seconds": f"{statistics.median(training_times):.6f}",
                "training_peak_rss_median_mb": f"{statistics.median(training_peaks) / 1_048_576:.3f}",
                "inference_samples": INFERENCE_RUNS,
                "inference_median_milliseconds": (
                    f"{statistics.median(inference_times) * 1000:.6f}"
                ),
                "inference_peak_rss_mb": f"{inference_peak_rss / 1_048_576:.3f}",
                "model_size_bytes": model_bytes,
                "model_size_kb": f"{model_bytes / 1024:.3f}",
            }
        )

    bundle = RESULTS_DIR / "model.joblib"
    joblib.dump(models, bundle)

    with (RESULTS_DIR / "baseline_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=["model", "test_accuracy"])
        writer.writeheader()
        writer.writerows(accuracy_rows)

    measurement_fields = [
        "model",
        "training_runs",
        "training_median_seconds",
        "training_peak_rss_median_mb",
        "inference_samples",
        "inference_median_milliseconds",
        "inference_peak_rss_mb",
        "model_size_bytes",
        "model_size_kb",
    ]
    with (RESULTS_DIR / "system_measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=measurement_fields)
        writer.writeheader()
        writer.writerows(measurement_rows)

    print(f"Results written to {RESULTS_DIR}")
    print(f"Combined model bundle size: {bundle.stat().st_size} bytes")


if __name__ == "__main__":
    main()
