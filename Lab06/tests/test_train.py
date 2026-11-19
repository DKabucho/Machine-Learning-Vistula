import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Lab06.src.train import train_epoch


def test_gradient_accumulation_matches_effective_batch() -> None:
    torch.manual_seed(7)
    images = torch.randn(8, 4)
    labels = torch.randint(0, 3, (8,))
    dataset = TensorDataset(images, labels)
    batched_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    microbatch_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    large_batch_model = nn.Linear(4, 3)
    accumulated_model = copy.deepcopy(large_batch_model)
    large_batch_optimizer = torch.optim.SGD(large_batch_model.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)

    train_epoch(
        large_batch_model,
        batched_loader,
        large_batch_optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        accumulation_steps=1,
    )
    train_epoch(
        accumulated_model,
        microbatch_loader,
        accumulated_optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        accumulation_steps=4,
    )

    for large_parameter, accumulated_parameter in zip(
        large_batch_model.parameters(), accumulated_model.parameters()
    ):
        torch.testing.assert_close(large_parameter, accumulated_parameter)
