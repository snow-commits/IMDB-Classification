from __future__ import annotations

import torch


def select_device() -> torch.device:
    """Select CUDA first, then MPS, then CPU for local training and evaluation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_enabled_for_device(requested_amp: bool, device: torch.device) -> bool:
    """Enable AMP only on CUDA devices for this project."""
    return requested_amp and device.type == "cuda"
