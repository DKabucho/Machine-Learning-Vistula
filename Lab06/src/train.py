from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import yaml
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

LAB03_SRC = Path(__file__).resolve().parents[2] / "Lab03" / "src"
sys.path.insert(0, str(LAB03_SRC))
from models import CNN  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def seed_worker(worker_id: int) -> None:
    worker_seed_value = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed_value)
    random.seed(worker_seed_value)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    config["_config_path"] = str(path.resolve())
    config["_project_dir"] = str(path.resolve().parent)
    return config


def resolved_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config[key])
    return value if value.is_absolute() else Path(config["_project_dir"]) / value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    data_dir = resolved_path(config, "data_dir")
    training = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=bool(config["download"]),
        transform=transform,
    )
    test = datasets.FashionMNIST(
        root=str(data_dir),
        train=False,
        download=bool(config["download"]),
        transform=transform,
    )
    train_set, validation_set = random_split(
        training,
        [int(config["train_size"]), int(config["validation_size"])],
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    workers = int(config["num_workers"])
    pin_memory = str(config["device"]).lower() != "cpu" and torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, validation_loader, test_loader


def select_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["device"])
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    common = {
        "lr": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
    }
    optimizer_name = str(config["optimizer"]).lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), momentum=float(config["momentum"]), **common
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **common)
    raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def make_scheduler(
    config: dict[str, Any], optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = str(config["scheduler"]).lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config["step_size"]),
            gamma=float(config["gamma"]),
        )
    if scheduler_name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["epochs"]),
            eta_min=float(config["minimum_learning_rate"]),
        )
    if scheduler_name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=float(config["gamma"]), patience=2
        )
    raise ValueError(f"Unsupported scheduler: {config['scheduler']}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accumulation_steps: int,
    profiler: Any = None,
    memory_probe: Callable[[], None] | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    optimizer.zero_grad(set_to_none=True)
    batches = len(loader)

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_end = min(group_start + accumulation_steps, batches)
        with record_function("forward_pass"):
            logits = model(images)
            loss = criterion(logits, labels)
        with record_function("backward_pass"):
            (loss * images.shape[0]).backward()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

        if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == batches:
            group_examples = sum(
                min(int(loader.batch_size), len(loader.dataset) - index * int(loader.batch_size))
                for index in range(group_start, group_end)
            )
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_examples)
            with record_function("optimizer_step"):
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if profiler is not None:
            profiler.step()
        if memory_probe is not None:
            memory_probe()

    return total_loss / seen, correct / seen


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    history: list[dict[str, float]],
    config: dict[str, Any],
    data_loader_generator_state: torch.Tensor,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "history": history,
            "config": {
                key: value
                for key, value in config.items()
                if not key.startswith("_")
            },
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "data_loader_generator_state": data_loader_generator_state,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> tuple[int, list[dict[str, float]], torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        checkpoint["history"],
        checkpoint["data_loader_generator_state"],
    )


def run_training(
    config: dict[str, Any],
    run_name: str,
    resume: Path | None = None,
    stop_after: int | None = None,
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = select_device(config)
    train_loader, validation_loader, test_loader = make_loaders(config)
    model = CNN().to(device)
    optimizer = make_optimizer(config, model)
    scheduler = make_scheduler(config, optimizer)
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = resolved_path(config, "checkpoint_dir")
    checkpoint_path = checkpoint_dir / f"{run_name.replace('/', '-')}-last.pt"
    start_epoch = 0
    history: list[dict[str, float]] = []
    loader_generator_state = train_loader.generator.get_state()
    if resume is not None:
        start_epoch, history, loader_generator_state = load_checkpoint(
            resume, model, optimizer, scheduler, device
        )
        train_loader.generator.set_state(loader_generator_state)

    output_dir = resolved_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(resolved_path(config, "tracking_uri").as_uri())
    mlflow.set_experiment(str(config["experiment_name"]))
    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        mlflow.log_params(
            {
                key: value
                for key, value in config.items()
                if not key.startswith("_") and not isinstance(value, (dict, list))
            }
        )
        mlflow.log_param("device_used", str(device))
        mlflow.log_param("resume_from", str(resume) if resume else "none")
        mlflow.log_artifact(config["_config_path"], artifact_path="config")

        last_epoch = min(int(config["epochs"]), stop_after or int(config["epochs"]))
        for epoch in range(start_epoch, last_epoch):
            epoch_started = time.perf_counter()
            train_loss, train_accuracy = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                int(config["gradient_accumulation_steps"]),
            )
            train_seconds = time.perf_counter() - epoch_started
            validation_started = time.perf_counter()
            validation_loss, validation_accuracy = evaluate(
                model, validation_loader, device
            )
            validation_seconds = time.perf_counter() - validation_started
            if isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(validation_loss)
            elif scheduler is not None:
                scheduler.step()
            metrics = {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": train_seconds,
                "validation_seconds": validation_seconds,
            }
            history.append(metrics)
            mlflow.log_metrics(
                {key: value for key, value in metrics.items() if key != "epoch"},
                step=epoch + 1,
            )
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                history,
                config,
                train_loader.generator.get_state(),
            )
            print(
                f"{run_name} epoch {epoch + 1}/{last_epoch}: "
                f"loss={train_loss:.4f}, val_acc={validation_accuracy:.4f}"
            )

        test_loss, test_accuracy = evaluate(model, test_loader, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
        history_path = output_dir / f"{run_name.replace('/', '-')}-history.csv"
        write_csv(history_path, history)
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(history_path), artifact_path="metrics")
        return {
            "run": run_name,
            "epochs": last_epoch,
            "optimizer": config["optimizer"],
            "scheduler": config["scheduler"],
            "batch_size": config["batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "num_workers": config["num_workers"],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "history": history,
            "checkpoint": str(checkpoint_path),
            "epoch_seconds": (
                history[-1]["epoch_seconds"] + history[-1]["validation_seconds"]
                if history
                else 0.0
            ),
            "validation_seconds": history[-1]["validation_seconds"] if history else 0.0,
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timed_profile(config: dict[str, Any]) -> dict[str, float]:
    seed_everything(int(config["seed"]))
    device = select_device(config)
    train_loader, _, _ = make_loaders(config)
    model = CNN().to(device)
    optimizer = make_optimizer(config, model)
    criterion = nn.CrossEntropyLoss()
    iterator = iter(train_loader)
    totals = {"data_loading": 0.0, "forward_pass": 0.0, "backward_pass": 0.0, "optimizer_step": 0.0}
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    output_dir = resolved_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=5, repeat=1),
    ) as profiler:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(len(train_loader)):
            sync()
            start = time.perf_counter()
            with record_function("data_loading"):
                images, labels = next(iterator)
                images = images.to(device)
                labels = labels.to(device)
            sync()
            totals["data_loading"] += time.perf_counter() - start
            start = time.perf_counter()
            with record_function("forward_pass"):
                logits = model(images)
                loss = criterion(logits, labels)
            sync()
            totals["forward_pass"] += time.perf_counter() - start

            start = time.perf_counter()
            with record_function("backward_pass"):
                loss.backward()
            sync()
            totals["backward_pass"] += time.perf_counter() - start

            start = time.perf_counter()
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            sync()
            totals["optimizer_step"] += time.perf_counter() - start
            profiler.step()

    profiler.export_chrome_trace(
        str(resolved_path(config, "output_dir") / "profile-trace.json")
    )
    total = sum(totals.values())
    result = {
        stage: seconds for stage, seconds in totals.items()
    }
    result.update(
        {f"{stage}_percent": seconds / total * 100 for stage, seconds in totals.items()}
    )
    result["total_seconds"] = total
    result["device"] = str(device)

    write_csv(output_dir / "profile.csv", [result])
    with (output_dir / "profile.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    mlflow.set_tracking_uri(resolved_path(config, "tracking_uri").as_uri())
    mlflow.set_experiment(str(config["experiment_name"]))
    with mlflow.start_run(run_name="one-epoch-profile"):
        mlflow.log_params({"seed": config["seed"], "num_workers": config["num_workers"], "batch_size": config["batch_size"]})
        mlflow.log_metrics({key: value for key, value in result.items() if isinstance(value, (int, float))})
        for artifact in ("profile.csv", "profile.json", "profile-trace.json"):
            mlflow.log_artifact(str(output_dir / artifact), artifact_path="profile")
    print("One-epoch stage shares:")
    for stage in totals:
        print(f"  {stage}: {result[f'{stage}_percent']:.2f}%")
    return result


def run_sweep(
    base_config: dict[str, Any], name: str, variants: list[dict[str, Any]], epochs: int
) -> list[dict[str, Any]]:
    results = []
    for variant in variants:
        config = copy.deepcopy(base_config)
        config.update(variant)
        config["epochs"] = epochs
        run_name = name + "-" + "-".join(f"{key}{value}" for key, value in variant.items())
        results.append(run_training(config, run_name))
    return results


def save_sweep(
    config: dict[str, Any], filename: str, results: list[dict[str, Any]], x_key: str, y_key: str, xlabel: str, ylabel: str
) -> None:
    output_dir = resolved_path(config, "output_dir")
    rows = [{key: value for key, value in result.items() if key != "history"} for result in results]
    csv_path = output_dir / filename
    write_csv(csv_path, rows)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([row[x_key] for row in rows], [row[y_key] for row in rows], marker="o")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plot_path = output_dir / filename.replace(".csv", ".png")
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    mlflow.set_tracking_uri(resolved_path(config, "tracking_uri").as_uri())
    mlflow.set_experiment(str(config["experiment_name"]))
    with mlflow.start_run(run_name=filename.removesuffix(".csv") + "-summary"):
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(plot_path))


def run_workers(config: dict[str, Any]) -> None:
    workers = [0, 1, 2, 4, 8]
    results = run_sweep(
        config, "workers", [{"num_workers": count} for count in workers], epochs=1
    )
    save_sweep(
        config,
        "worker-sweep.csv",
        results,
        "num_workers",
        "epoch_seconds",
        "Data-loader workers",
        "Epoch time (s)",
    )


def run_batch_sizes(config: dict[str, Any]) -> None:
    results = []
    for batch_size in [32, 64, 128, 256, 512]:
        run_config = copy.deepcopy(config)
        run_config["batch_size"] = batch_size
        run_config["epochs"] = 5
        seed_everything(int(run_config["seed"]))
        device = select_device(run_config)
        train_loader, validation_loader, test_loader = make_loaders(run_config)
        model = CNN().to(device)
        optimizer = make_optimizer(run_config, model)
        criterion = nn.CrossEntropyLoss()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        process = __import__("psutil").Process()
        peak_memory = process.memory_info().rss

        def sample_memory() -> None:
            nonlocal peak_memory
            peak_memory = max(peak_memory, process.memory_info().rss)

        training_seconds = 0.0
        epochs_history = []
        for epoch in range(5):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_started = time.perf_counter()
            train_loss, train_accuracy = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                1,
                memory_probe=sample_memory,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds += time.perf_counter() - epoch_started
            validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
            epochs_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_memory_mb = peak_memory / (1024**2)
        _, test_accuracy = evaluate(model, test_loader, device)
        throughput = len(train_loader.dataset) * 5 / training_seconds
        result = {
            "batch_size": batch_size,
            "epochs": 5,
            "throughput_samples_per_second": throughput,
            "peak_memory_mb": peak_memory_mb,
            "test_accuracy": test_accuracy,
            "test_accuracy_percent": test_accuracy * 100,
            "device": str(device),
        }
        results.append(result)
        mlflow.set_tracking_uri(resolved_path(run_config, "tracking_uri").as_uri())
        mlflow.set_experiment(str(run_config["experiment_name"]))
        with mlflow.start_run(run_name=f"batch-size-{batch_size}"):
            mlflow.log_params({"batch_size": batch_size, "epochs": 5, "seed": config["seed"]})
            mlflow.log_metrics({key: value for key, value in result.items() if isinstance(value, (int, float))})
            mlflow.log_dict(epochs_history, "metrics/epoch-history.json")
        print(f"batch={batch_size}: {throughput:.1f} samples/s, {peak_memory_mb:.1f} MB, test_acc={test_accuracy:.4f}")
    save_sweep(
        config,
        "batch-size-sweep.csv",
        [{**result, "history": []} for result in results],
        "batch_size",
        "throughput_samples_per_second",
        "Batch size",
        "Throughput (samples/s)",
    )


def run_optimizer_scheduler(config: dict[str, Any]) -> None:
    variants = [
        {"optimizer": optimizer, "scheduler": scheduler}
        for optimizer in ["SGD", "Adam"]
        for scheduler in ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    ]
    results = run_sweep(config, "optimizer-scheduler", variants, epochs=10)
    save_sweep(
        config,
        "optimizer-scheduler.csv",
        results,
        "run",
        "test_accuracy",
        "Optimizer / scheduler configuration",
        "Test accuracy",
    )


def run_accumulation(config: dict[str, Any]) -> None:
    results = run_sweep(
        config,
        "accumulation",
        [
            {"batch_size": 256, "gradient_accumulation_steps": 1},
            {"batch_size": 64, "gradient_accumulation_steps": 4},
        ],
        epochs=5,
    )
    save_sweep(
        config,
        "gradient-accumulation.csv",
        results,
        "run",
        "test_accuracy",
        "Training configuration",
        "Test accuracy",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Lab 6 CNN training and experiments")
    parser.add_argument(
        "experiment",
        choices=["train", "profile", "workers", "batch-sizes", "optimizers", "accumulation"],
        nargs="?",
        default="train",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be at least 1")
    if args.experiment == "train":
        run_training(config, "cnn-training", args.resume, args.stop_after)
    elif args.experiment == "profile":
        timed_profile(config)
    elif args.experiment == "workers":
        run_workers(config)
    elif args.experiment == "batch-sizes":
        run_batch_sizes(config)
    elif args.experiment == "optimizers":
        run_optimizer_scheduler(config)
    else:
        run_accumulation(config)


if __name__ == "__main__":
    main()
