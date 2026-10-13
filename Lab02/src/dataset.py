from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels = load_digits(return_X_y=True)
    features = features / 16.0
    one_hot_labels = np.eye(10, dtype=np.float64)[labels]

    return train_test_split(
        features,
        one_hot_labels,
        test_size=0.30,
        stratify=labels,
        random_state=42,
    )


def main() -> None:
    x_train, x_test, y_train, y_test = prepare_data()
    print(f"Training features: {x_train.shape}")
    print(f"Test features: {x_test.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test labels: {y_test.shape}")


if __name__ == "__main__":
    main()
