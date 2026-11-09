"""Run Lab05 gradient checks, training benchmarks, and PyTorch inspections."""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import sklearn
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import nn

from src.autodiff import Value


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "lab05_results.json"


def check_gradients() -> dict[str, object]:
    values = (0.7, -1.2, 0.4)

    def scalar_expression(a: float, b: float, c: float) -> float:
        return max((a * b + c) ** 2, 0.0)

    a, b, c = (Value(value) for value in values)
    output = ((a * b + c) ** 2).relu().sum()
    output.backward()
    autodiff_grads = [a.grad, b.grad, c.grad]

    torch_values = [
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in values
    ]
    torch_output = (
        (torch_values[0] * torch_values[1] + torch_values[2]) ** 2
    ).relu().sum()
    torch_output.backward()
    torch_grads = [float(value.grad) for value in torch_values]

    step = 1e-6
    numerical_grads = []
    for index, value in enumerate(values):
        plus = list(values)
        minus = list(values)
        plus[index] += step
        minus[index] -= step
        numerical_grads.append(
            (scalar_expression(*plus) - scalar_expression(*minus)) / (2 * step)
        )

    torch_difference = max(
        abs(left - right) for left, right in zip(autodiff_grads, torch_grads)
    )
    numerical_difference = max(
        abs(left - right) for left, right in zip(autodiff_grads, numerical_grads)
    )
    if max(torch_difference, numerical_difference) >= 1e-6:
        raise AssertionError(
            "Gradient check failed: "
            f"PyTorch difference={torch_difference}, "
            f"numerical difference={numerical_difference}"
        )
    return {
        "expression": "((a*b + c)**2).relu().sum()",
        "input_values": list(values),
        "output": output.data,
        "autodiff_gradients": autodiff_grads,
        "pytorch_gradients": torch_grads,
        "numerical_gradients": numerical_grads,
        "max_abs_difference_vs_pytorch": torch_difference,
        "max_abs_difference_vs_numerical": numerical_difference,
        "tolerance": 1e-6,
        "passed": True,
    }


class SmallMoonNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.activation = nn.ReLU()
        self.output = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output(self.activation(self.hidden(inputs))))


def _scalar_model_parameters(rng: np.random.Generator) -> list[list[Value]]:
    parameters: list[list[Value]] = []
    for fan_in, fan_out in ((2, 8), (8, 1)):
        weights = [
            Value(float(value))
            for value in rng.normal(0.0, 0.45, size=(fan_out, fan_in)).flat
        ]
        biases = [Value(0.0) for _ in range(fan_out)]
        parameters.extend([weights, biases])
    return parameters


def _scalar_forward(
    inputs: list[float], parameters: list[list[Value]]
) -> Value:
    hidden_weights, hidden_biases, output_weights, output_biases = parameters
    hidden = [
        sum(
            (hidden_weights[index * 2 + feature] * inputs[feature]
             for feature in range(2)),
            hidden_biases[index],
        ).relu()
        for index in range(8)
    ]
    logit = sum(
        (output_weights[index] * hidden[index] for index in range(8)),
        output_biases[0],
    )
    return logit.sigmoid()


def _scalar_train(
    features: np.ndarray, labels: np.ndarray, seed: int, epochs: int
) -> float:
    parameters = _scalar_model_parameters(np.random.default_rng(seed))
    learning_rate = 0.08
    for _ in range(epochs):
        losses = []
        for row, label in zip(features, labels):
            probability = _scalar_forward(row.tolist(), parameters)
            target = float(label)
            losses.append(
                -(target * probability.log()
                  + (1.0 - target) * (1.0 - probability).log())
            )
        loss = sum(losses, Value(0.0)) / len(losses)
        loss.backward()
        for group in parameters:
            for parameter in group:
                parameter.data -= learning_rate * parameter.grad
                parameter.grad = 0.0
    return float(loss.data)


def _torch_train(
    features: torch.Tensor, labels: torch.Tensor, seed: int, epochs: int
) -> float:
    torch.manual_seed(seed)
    model = SmallMoonNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    loss_function = nn.BCELoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), labels)
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def _median_runtime(function: Callable[[], float], repeats: int = 5) -> dict[str, object]:
    durations = []
    final_losses = []
    for repeat in range(repeats):
        started = time.perf_counter()
        final_losses.append(function())
        durations.append(time.perf_counter() - started)
    return {
        "median_seconds": statistics.median(durations),
        "run_seconds": durations,
        "final_loss_per_run": final_losses,
        "repeats": repeats,
    }


def benchmark_training() -> dict[str, object]:
    features, labels = make_moons(n_samples=192, noise=0.2, random_state=17)
    x_train, _, y_train, _ = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=17,
        stratify=labels,
    )
    epochs = 10
    torch_features = torch.tensor(x_train, dtype=torch.float32)
    torch_labels = torch.tensor(y_train[:, None], dtype=torch.float32)
    _scalar_train(x_train, y_train, seed=5, epochs=epochs)
    _torch_train(torch_features, torch_labels, seed=5, epochs=epochs)
    manual = _median_runtime(
        lambda: _scalar_train(x_train, y_train, seed=5, epochs=epochs)
    )
    pytorch = _median_runtime(
        lambda: _torch_train(torch_features, torch_labels, seed=5, epochs=epochs)
    )
    manual["examples_per_epoch"] = len(x_train)
    pytorch["examples_per_epoch"] = len(x_train)
    return {
        "dataset": "sklearn.datasets.make_moons",
        "samples": len(features),
        "train_samples": len(x_train),
        "architecture": "2 -> 8 -> 1 (ReLU hidden, sigmoid output)",
        "loss": "binary cross entropy",
        "epochs": epochs,
        "optimizer": "SGD, learning rate 0.08",
        "device": "CPU",
        "custom_scalar_autodiff": manual,
        "pytorch": pytorch,
        "speedup_pytorch_over_custom": (
            manual["median_seconds"] / pytorch["median_seconds"]
        ),
    }


def _median_operation_time(
    operation: Callable[[], torch.Tensor], repeats: int = 5, iterations: int = 200
) -> dict[str, object]:
    for _ in range(10):
        operation()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        samples.append((time.perf_counter() - started) / iterations)
    return {
        "median_seconds_per_operation": statistics.median(samples),
        "sample_seconds_per_operation": samples,
        "repeats": repeats,
        "iterations_per_repeat": iterations,
    }


def inspect_torch() -> dict[str, object]:
    torch.manual_seed(23)
    model = SmallMoonNet().eval()
    example = torch.randn(64, 2)
    traced = torch.jit.trace(model, example)

    with torch.no_grad():
        eager_result = model(example)
        traced_result = traced(example)
    trace_difference = float((eager_result - traced_result).abs().max())
    if trace_difference >= 1e-6:
        raise AssertionError(f"Traced model differs from eager: {trace_difference}")

    for _ in range(20):
        with torch.no_grad():
            model(example)
            traced(example)

    def time_model(callable_model: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, object]:
        return _median_operation_time(
            lambda: callable_model(example), repeats=5, iterations=200
        )

    eager_time = time_model(model)
    traced_time = time_model(traced)
    fx_graph = str(torch.fx.symbolic_trace(model).graph)

    base = torch.randn(512, 512)
    non_contiguous = base.t()
    contiguous = non_contiguous.contiguous()
    if non_contiguous.is_contiguous() or not contiguous.is_contiguous():
        raise AssertionError("Expected transposed view and contiguous copy.")
    if not torch.equal(non_contiguous, contiguous):
        raise AssertionError("Contiguous copy changed tensor values.")
    non_contiguous_time = _median_operation_time(
        lambda: torch.relu(non_contiguous * 1.01), iterations=100
    )
    contiguous_time = _median_operation_time(
        lambda: torch.relu(contiguous * 1.01), iterations=100
    )
    non_contiguous_time["layout"] = "transpose view"
    contiguous_time["layout"] = "contiguous copy of transpose"

    return {
        "graph_inspection": {
            "method": "torch.fx.symbolic_trace",
            "graph": fx_graph,
            "model": "Linear(2, 8) -> ReLU -> Linear(8, 1) -> sigmoid",
        },
        "execution_comparison": {
            "compiled_or_traced": "torch.jit.trace",
            "warmup_calls_per_mode": 20,
            "input_shape": list(example.shape),
            "input_dtype": str(example.dtype),
            "input_contiguous": example.is_contiguous(),
            "max_abs_output_difference": trace_difference,
            "eager": eager_time,
            "traced": traced_time,
            "eager_over_traced_speed_ratio": (
                eager_time["median_seconds_per_operation"]
                / traced_time["median_seconds_per_operation"]
            ),
        },
        "layout_comparison": {
            "base_shape": list(base.shape),
            "view_shape": list(non_contiguous.shape),
            "view_dtype": str(non_contiguous.dtype),
            "view_stride": list(non_contiguous.stride()),
            "view_contiguous": non_contiguous.is_contiguous(),
            "contiguous_stride": list(contiguous.stride()),
            "contiguous_tensor_contiguous": contiguous.is_contiguous(),
            "operation": "relu(tensor * 1.01)",
            "non_contiguous": non_contiguous_time,
            "contiguous": contiguous_time,
            "non_contiguous_over_contiguous_speed_ratio": (
                non_contiguous_time["median_seconds_per_operation"]
                / contiguous_time["median_seconds_per_operation"]
            ),
        },
    }


def main() -> None:
    torch.set_num_threads(1)
    results = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "pytorch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "gradient_validation": check_gradients(),
        "training_benchmark": benchmark_training(),
        "pytorch_inspection_and_benchmarks": inspect_torch(),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
