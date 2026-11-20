from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAB03_SRC = REPOSITORY_ROOT / "Lab03" / "src"
sys.path.insert(0, str(LAB03_SRC))
from models import CNN  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


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


def load_test_loader(config: dict[str, Any]) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    test_set = datasets.FashionMNIST(
        root=str(resolve_path(config, "data_dir")),
        train=False,
        download=False,
        transform=transform,
    )
    return DataLoader(test_set, batch_size=int(config["batch_size"]), shuffle=False)


@torch.inference_mode()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    examples = 0
    for images, labels in loader:
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        examples += labels.size(0)
    return correct / examples


def serialized_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024**2)


@torch.inference_mode()
def latency_ms(
    model: nn.Module,
    sample: torch.Tensor,
    warmup_runs: int,
    timed_runs: int,
) -> tuple[float, float]:
    model.eval()
    for _ in range(warmup_runs):
        model(sample)

    durations = []
    for _ in range(timed_runs):
        started = time.perf_counter_ns()
        model(sample)
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    durations.sort()
    median = float(np.median(durations))
    p95_index = min(len(durations) - 1, int(np.ceil(0.95 * len(durations))) - 1)
    return median, durations[p95_index]


def make_model(checkpoint_path: Path) -> CNN:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the Lab 6 baseline first."
        )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = CNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def compare(config: dict[str, Any]) -> list[dict[str, Any]]:
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.set_num_threads(1)

    test_loader = load_test_loader(config)
    float_model = make_model(resolve_path(config, "checkpoint"))
    int8_model = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(float_model),
        {nn.Linear},
        dtype=torch.qint8,
    ).eval()
    latency_loader = DataLoader(
        test_loader.dataset, batch_size=int(config["latency_batch_size"]), shuffle=False
    )
    sample, _ = next(iter(latency_loader))

    measurements = []
    for name, model in (("float32", float_model), ("dynamic_int8", int8_model)):
        median, p95 = latency_ms(
            model,
            sample,
            int(config["warmup_runs"]),
            int(config["timed_runs"]),
        )
        test_accuracy = accuracy(model, test_loader)
        size_mb = serialized_size_mb(model)
        measurements.append(
            {
                "model": name,
                "test_accuracy": test_accuracy,
                "test_accuracy_percent": test_accuracy * 100,
                "batch1_median_latency_ms": median,
                "batch1_p95_latency_ms": p95,
                "state_dict_size_mb": size_mb,
                "accuracy_target_pass": test_accuracy
                >= float(config["accuracy_target"]),
                "latency_target_pass": p95
                <= float(config["latency_target_ms"]),
                "size_target_pass": size_mb <= float(config["size_target_mb"]),
            }
        )
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the Fashion-MNIST CNN baseline with dynamic int8 quantization."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    rows = compare(config)

    output_dir = resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quantization-comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "quantization-comparison.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(rows, file, indent=2)

    for row in rows:
        print(
            f"{row['model']}: accuracy={row['test_accuracy_percent']:.2f}%, "
            f"batch-1 median/p95={row['batch1_median_latency_ms']:.3f}/"
            f"{row['batch1_p95_latency_ms']:.3f} ms, "
            f"state-dict={row['state_dict_size_mb']:.4f} MB"
        )
        print(
            "  targets: "
            f"accuracy={'PASS' if row['accuracy_target_pass'] else 'FAIL'}, "
            f"latency={'PASS' if row['latency_target_pass'] else 'FAIL'}, "
            f"size={'PASS' if row['size_target_pass'] else 'FAIL'}"
        )
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
