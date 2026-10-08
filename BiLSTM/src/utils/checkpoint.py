from pathlib import Path

import torch


def save_checkpoint(
    checkpoint: dict[str, object],
    checkpoint_path: str | Path,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, object]:
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary, got: {type(checkpoint)}")

    return checkpoint