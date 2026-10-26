from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from cost_analysis import median_inference_ms
from dataset import BATCH_SIZE, create_dataloaders
from models import CNN, MLP, PatchTransformer


EPOCHS = 5
LEARNING_RATE = 1e-3
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def evaluate(
    model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_one_model(
    name: str,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, int | float | str]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_samples = labels.size(0)
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

        print(
            f"{name} epoch {epoch}/{EPOCHS}: "
            f"train loss={total_loss / total_samples:.4f}"
        )

    test_accuracy = evaluate(model, test_loader, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    # Benchmark on CPU for consistency with cost_analysis.py measurements.
    model = model.to("cpu").eval()
    batch_1_latency = median_inference_ms(model, batch_size=1, runs=100)
    batch_128_latency = median_inference_ms(model, batch_size=BATCH_SIZE, runs=20)
    row: dict[str, int | float | str] = {
        "model": name,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "parameters": parameter_count,
        "test_accuracy": test_accuracy,
        "test_accuracy_percent": test_accuracy * 100,
        "batch_1_median_ms_100_runs": batch_1_latency,
        "batch_128_median_ms_20_runs": batch_128_latency,
    }
    print(
        f"{name}: test accuracy={test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%), parameters={parameter_count:,}, "
        f"CPU latency batch 1={batch_1_latency:.6f} ms, "
        f"batch 128={batch_128_latency:.6f} ms"
    )
    return row


def create_plots(rows: list[dict[str, int | float | str]]) -> None:
    names = [str(row["model"]) for row in rows]
    accuracy = [float(row["test_accuracy_percent"]) for row in rows]

    figure, axis = plt.subplots(figsize=(8, 5))
    for row, name, score in zip(rows, names, accuracy):
        parameters_thousands = int(row["parameters"]) / 1_000
        axis.scatter(parameters_thousands, score, s=70)
        axis.annotate(name, (parameters_thousands, score), xytext=(6, 6), textcoords="offset points")
    axis.set_xlabel("Trainable parameters (thousands)")
    axis.set_ylabel("Official test accuracy (%)")
    axis.set_title("Fashion-MNIST accuracy vs. model parameters")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "accuracy_vs_parameters.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for row, name, score in zip(rows, names, accuracy):
        latency_ms = float(row["batch_1_median_ms_100_runs"])
        axis.scatter(latency_ms, score, s=70)
        axis.annotate(name, (latency_ms, score), xytext=(6, 6), textcoords="offset points")
    axis.set_xlabel("CPU batch-1 inference latency (ms, median of 100 runs)")
    axis.set_ylabel("Official test accuracy (%)")
    axis.set_title("Fashion-MNIST accuracy vs. inference latency")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "accuracy_vs_latency.png", dpi=150)
    plt.close(figure)


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, _, test_loader = create_dataloaders(batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    print(f"Training configuration: {EPOCHS} epochs, Adam, lr={LEARNING_RATE}, batch={BATCH_SIZE}")

    model_factories = {
        "MLP": MLP,
        "CNN": CNN,
        "PatchTransformer": PatchTransformer,
    }
    rows = []
    for index, (name, model_factory) in enumerate(model_factories.items()):
        model_seed = SEED + index
        random.seed(model_seed)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        rows.append(
            train_one_model(
                name,
                model_factory(),
                train_loader,
                test_loader,
                device,
            )
        )

    csv_path = RESULTS_DIR / "training_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    create_plots(rows)
    print(f"Saved results to {csv_path}")
    print(f"Saved plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
