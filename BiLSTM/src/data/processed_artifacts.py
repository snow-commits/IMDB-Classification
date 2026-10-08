from __future__ import annotations

import json
from pathlib import Path
from typing import Final, TypedDict

from src.data.dataset import LabeledExample, UnlabeledExample
from src.data.vocab import Vocab, load_vocab

DEFAULT_VOCAB_OUTPUT_PATH: Final[str] = "data/vocab.json"
DEFAULT_TRAIN_PROCESSED_PATH: Final[str] = "data/processed/train_processed.json"
DEFAULT_TEST_PROCESSED_PATH: Final[str] = "data/processed/test_processed.json"
DEFAULT_UNSUPERVISED_PROCESSED_PATH: Final[str] = "data/processed/unsupervised_processed.json"


class PreprocessingArtifacts(TypedDict):
    vocab: Vocab
    train_examples: list[LabeledExample]
    test_examples: list[LabeledExample]
    unsupervised_examples: list[UnlabeledExample]


def save_processed_supervised_examples(
    examples: list[LabeledExample],
    output_path: str | Path,
) -> None:
    """Persist labeled processed examples to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(examples, file_handle, ensure_ascii=False)


def save_processed_unsupervised_examples(
    examples: list[UnlabeledExample],
    output_path: str | Path,
) -> None:
    """Persist unlabeled processed examples to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(examples, file_handle, ensure_ascii=False)


def load_processed_supervised_examples(processed_path: str | Path) -> list[LabeledExample]:
    """Load labeled processed examples from JSON."""
    path = Path(processed_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed supervised file not found: {path}")
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, list):
        raise TypeError("Processed supervised payload must be a list.")
    return payload


def load_processed_unsupervised_examples(processed_path: str | Path) -> list[UnlabeledExample]:
    """Load unlabeled processed examples from JSON."""
    path = Path(processed_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed unsupervised file not found: {path}")
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, list):
        raise TypeError("Processed unsupervised payload must be a list.")
    return payload


def load_processed_artifacts_from_config(config: dict[str, object]) -> PreprocessingArtifacts:
    """Load vocab and processed example artifacts from config-defined paths."""
    data_config = config["data"]
    vocab_config = config["vocab"]

    if not isinstance(data_config, dict):
        raise TypeError("config['data'] must be a dictionary.")
    if not isinstance(vocab_config, dict):
        raise TypeError("config['vocab'] must be a dictionary.")

    return {
        "vocab": load_vocab(str(vocab_config.get("output_path", DEFAULT_VOCAB_OUTPUT_PATH))),
        "train_examples": load_processed_supervised_examples(
            str(data_config.get("train_processed_path", DEFAULT_TRAIN_PROCESSED_PATH))
        ),
        "test_examples": load_processed_supervised_examples(
            str(data_config.get("test_processed_path", DEFAULT_TEST_PROCESSED_PATH))
        ),
        "unsupervised_examples": load_processed_unsupervised_examples(
            str(data_config.get("unsupervised_processed_path", DEFAULT_UNSUPERVISED_PROCESSED_PATH))
        ),
    }
