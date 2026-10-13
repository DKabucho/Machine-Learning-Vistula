from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from dataset import prepare_data


LEARNING_RATES = (0.001, 0.01, 0.1)
HIDDEN_WIDTHS = (16, 64)
BATCH_SIZE = 32
EPOCHS = 50
TIMING_RUNS = 100
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


class DigitNetwork:
    def __init__(self, hidden_width: int, random_state: int = 42) -> None:
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")

        rng = np.random.default_rng(random_state)
        self.weights = (
            rng.normal(0.0, 0.01, size=(64, hidden_width)),
            rng.normal(0.0, 0.01, size=(hidden_width, hidden_width)),
            rng.normal(0.0, 0.01, size=(hidden_width, 10)),
        )
        self.biases = (
            np.zeros(hidden_width),
            np.zeros(hidden_width),
            np.zeros(10),
        )

    def forward(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != 64:
            raise ValueError("features must have shape (batch, 64)")

        return self._forward_with_cache(features)[-1]

    def _forward_with_cache(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pre_activation1 = features @ self.weights[0] + self.biases[0]
        hidden1 = np.maximum(pre_activation1, 0)
        pre_activation2 = hidden1 @ self.weights[1] + self.biases[1]
        hidden2 = np.maximum(pre_activation2, 0)
        logits = hidden2 @ self.weights[2] + self.biases[2]

        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return pre_activation1, hidden1, pre_activation2, hidden2, probabilities

    def gradients(
        self, features: np.ndarray, one_hot_labels: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        if features.ndim != 2 or features.shape[1] != 64 or features.shape[0] == 0:
            raise ValueError("features must have shape (non-empty batch, 64)")
        if one_hot_labels.shape != (features.shape[0], 10):
            raise ValueError("one_hot_labels must have shape (batch, 10)")

        pre_activation1, hidden1, pre_activation2, hidden2, probabilities = (
            self._forward_with_cache(features)
        )
        return self._backward_from_cache(
            features,
            one_hot_labels,
            (pre_activation1, hidden1, pre_activation2, hidden2, probabilities),
        )

    def _backward_from_cache(
        self,
        features: np.ndarray,
        one_hot_labels: np.ndarray,
        cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        pre_activation1, hidden1, pre_activation2, hidden2, probabilities = cache
        batch_size = features.shape[0]
        output_delta = (probabilities - one_hot_labels) / batch_size

        weight3_gradient = hidden2.T @ output_delta
        hidden2_delta = output_delta @ self.weights[2].T
        pre_activation2_delta = hidden2_delta * (pre_activation2 > 0)
        weight2_gradient = hidden1.T @ pre_activation2_delta
        hidden1_delta = pre_activation2_delta @ self.weights[1].T
        pre_activation1_delta = hidden1_delta * (pre_activation1 > 0)
        weight1_gradient = features.T @ pre_activation1_delta

        weight_gradients = (
            weight1_gradient,
            weight2_gradient,
            weight3_gradient,
        )
        bias_gradients = (
            np.sum(pre_activation1_delta, axis=0),
            np.sum(pre_activation2_delta, axis=0),
            np.sum(output_delta, axis=0),
        )
        return weight_gradients, bias_gradients

    def median_pass_times(
        self,
        features: np.ndarray,
        one_hot_labels: np.ndarray,
        runs: int = TIMING_RUNS,
    ) -> tuple[float, float]:
        cache = self._forward_with_cache(features)
        self.forward(features)
        self._backward_from_cache(features, one_hot_labels, cache)

        forward_times: list[float] = []
        backward_times: list[float] = []
        for _ in range(runs):
            started = perf_counter()
            self.forward(features)
            forward_times.append(perf_counter() - started)

            started = perf_counter()
            self._backward_from_cache(features, one_hot_labels, cache)
            backward_times.append(perf_counter() - started)

        return (
            float(np.median(forward_times) * 1000),
            float(np.median(backward_times) * 1000),
        )

    def train_batch(
        self, features: np.ndarray, one_hot_labels: np.ndarray, learning_rate: float
    ) -> None:
        weight_gradients, bias_gradients = self.gradients(features, one_hot_labels)
        for weight, gradient in zip(self.weights, weight_gradients):
            weight -= learning_rate * gradient
        for bias, gradient in zip(self.biases, bias_gradients):
            bias -= learning_rate * gradient


def cross_entropy_loss(
    probabilities: np.ndarray, one_hot_labels: np.ndarray
) -> float:
    if probabilities.shape != one_hot_labels.shape:
        raise ValueError("probabilities and labels must have the same shape")
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("inputs must have shape (non-empty batch, classes)")

    clipped_probabilities = np.clip(
        probabilities, np.finfo(probabilities.dtype).tiny, 1.0
    )
    return float(-np.sum(one_hot_labels * np.log(clipped_probabilities)) / len(probabilities))


def check_gradients(
    network: DigitNetwork,
    features: np.ndarray,
    one_hot_labels: np.ndarray,
    checks: int = 10,
    epsilon: float = 1e-5,
    random_state: int = 42,
) -> list[float]:
    weight_gradients, _ = network.gradients(features, one_hot_labels)
    total_weights = sum(weight.size for weight in network.weights)
    rng = np.random.default_rng(random_state)
    sampled_indices = rng.choice(total_weights, size=checks, replace=False)
    relative_errors: list[float] = []
    loss_at_weights = lambda: cross_entropy_loss(
        network.forward(features), one_hot_labels
    )

    layer_offsets = np.cumsum([0, *(weight.size for weight in network.weights)])
    for flat_index in sampled_indices:
        layer_index = int(np.searchsorted(layer_offsets[1:], flat_index, side="right"))
        local_index = int(flat_index - layer_offsets[layer_index])
        weight = network.weights[layer_index]
        analytic_slope = weight_gradients[layer_index].flat[local_index]
        original_value = weight.flat[local_index]

        try:
            weight.flat[local_index] = original_value + epsilon
            positive_loss = loss_at_weights()
            weight.flat[local_index] = original_value - epsilon
            negative_loss = loss_at_weights()
        finally:
            weight.flat[local_index] = original_value

        numerical_slope = (positive_loss - negative_loss) / (2 * epsilon)
        denominator = max(
            1e-8, abs(analytic_slope) + abs(numerical_slope)
        )
        relative_errors.append(
            float(abs(analytic_slope - numerical_slope) / denominator)
        )

    return relative_errors


def train_network(
    network: DigitNetwork,
    features: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
    random_state: int,
) -> list[float]:
    rng = np.random.default_rng(random_state)
    epoch_losses: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        shuffled_indices = rng.permutation(len(features))
        for start in range(0, len(features), BATCH_SIZE):
            batch_indices = shuffled_indices[start : start + BATCH_SIZE]
            network.train_batch(
                features[batch_indices], labels[batch_indices], learning_rate
            )

        epoch_loss = cross_entropy_loss(network.forward(features), labels)
        epoch_losses.append(epoch_loss)
        print(
            f"W={network.weights[0].shape[1]}, lr={learning_rate:g}, "
            f"epoch={epoch:02d}/{EPOCHS}, loss={epoch_loss:.6f}"
        )

    return epoch_losses


def main() -> None:
    x_train, x_test, y_train, y_test = prepare_data()
    sample = x_train[:8]
    label_sample = y_train[:8]

    for width in HIDDEN_WIDTHS:
        network = DigitNetwork(width)
        probabilities = network.forward(sample)
        if probabilities.shape != (len(sample), 10):
            raise RuntimeError(f"Unexpected output shape: {probabilities.shape}")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("Softmax output rows do not sum to 1")
        loss = cross_entropy_loss(probabilities, label_sample)
        if not np.isclose(loss, np.log(10), atol=0.01):
            raise RuntimeError(f"Unexpected untrained loss for W={width}: {loss}")
        print(
            f"W={width}: output shape {probabilities.shape}; "
            f"row sums verified; cross-entropy loss {loss:.4f}"
        )
        relative_errors = check_gradients(network, sample, label_sample)
        passing_checks = sum(error < 1e-6 for error in relative_errors)
        if passing_checks < 9:
            raise RuntimeError(
                f"Gradient check passed {passing_checks}/10 weights for W={width}"
            )
        print(
            f"W={width}: gradient check passed {passing_checks}/10 weights "
            f"(max relative error {max(relative_errors):.2e})"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    loss_curves: dict[tuple[int, float], list[float]] = {}
    accuracy_rows: list[dict[str, str | int]] = []

    for width in HIDDEN_WIDTHS:
        for learning_rate in LEARNING_RATES:
            network = DigitNetwork(width)
            losses = train_network(
                network, x_train, y_train, learning_rate, random_state=42
            )
            loss_curves[(width, learning_rate)] = losses
            predictions = np.argmax(network.forward(x_test), axis=1)
            expected = np.argmax(y_test, axis=1)
            accuracy = float(np.mean(predictions == expected))
            accuracy_rows.append(
                {
                    "hidden_width": width,
                    "learning_rate": learning_rate,
                    "epochs": EPOCHS,
                    "test_accuracy": f"{accuracy:.4f}",
                }
            )
            print(
                f"Final test accuracy: W={width}, lr={learning_rate:g}, "
                f"accuracy={accuracy:.4f}"
            )

    figure, axis = plt.subplots(figsize=(10, 6))
    for (width, learning_rate), losses in loss_curves.items():
        axis.plot(
            range(1, EPOCHS + 1),
            losses,
            label=f"W={width}, lr={learning_rate:g}",
        )
    axis.set(title="Training loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
    plt.close(figure)

    accuracy_path = RESULTS_DIR / "final_test_accuracy.csv"
    with accuracy_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["hidden_width", "learning_rate", "epochs", "test_accuracy"],
        )
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"Loss curves saved to {RESULTS_DIR / 'loss_curves.png'}")
    print(f"Final test accuracies saved to {accuracy_path}")

    timing_rows: list[dict[str, str | int]] = []
    timing_features = x_train[:BATCH_SIZE]
    timing_labels = y_train[:BATCH_SIZE]
    for width in HIDDEN_WIDTHS:
        network = DigitNetwork(width)
        forward_ms, backward_ms = network.median_pass_times(
            timing_features, timing_labels
        )
        timing_rows.append(
            {
                "hidden_width": width,
                "batch_size": BATCH_SIZE,
                "runs": TIMING_RUNS,
                "forward_median_ms": f"{forward_ms:.6f}",
                "backward_median_ms": f"{backward_ms:.6f}",
            }
        )
        dominant_pass = "backward" if backward_ms > forward_ms else "forward"
        print(
            f"Pass timing W={width}, batch={BATCH_SIZE}: "
            f"forward median={forward_ms:.6f} ms, "
            f"backward median={backward_ms:.6f} ms ({dominant_pass} dominates)"
        )

    timing_path = RESULTS_DIR / "pass_timings.csv"
    with timing_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "hidden_width",
                "batch_size",
                "runs",
                "forward_median_ms",
                "backward_median_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(timing_rows)
    print(f"Pass timings saved to {timing_path}")


if __name__ == "__main__":
    main()
