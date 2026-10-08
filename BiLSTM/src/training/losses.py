import torch
from torch import nn


def compute_supervised_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()
    return criterion(logits, labels)