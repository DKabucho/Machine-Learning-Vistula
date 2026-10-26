from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(torch.flatten(features, start_dim=1))


class PatchTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(7 * 7, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        patches = (
            images.reshape(batch_size, 1, 4, 7, 4, 7)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size, 16, 7 * 7)
        )
        embeddings = self.patch_projection(patches)
        encoded = self.encoder(embeddings)
        return self.classifier(encoded.mean(dim=1))


def main() -> None:
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "PatchTransformer": PatchTransformer(),
    }
    sample_images = torch.zeros(2, 1, 28, 28)
    for name, model in models.items():
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        output = model(sample_images)
        print(f"{name}: {parameter_count:,} parameters, output shape {tuple(output.shape)}")


if __name__ == "__main__":
    main()
