from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pandas as pd  # noqa: PANDAS_OK

from src.data.dataset import LabeledExample, UnlabeledExample
from src.data.processed_artifacts import (
    DEFAULT_TEST_PROCESSED_PATH,
    DEFAULT_TRAIN_PROCESSED_PATH,
    DEFAULT_UNSUPERVISED_PROCESSED_PATH,
    DEFAULT_VOCAB_OUTPUT_PATH,
    PreprocessingArtifacts,
    save_processed_supervised_examples,
    save_processed_unsupervised_examples,
)
from src.data.vocab import Vocab, build_vocab_from_token_lists, save_vocab, tokens_to_ids
from src.utils.config import load_and_validate_config

WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[^\w\s]")
DEFAULT_TEXT_COLUMN: Final[str] = "text"
DEFAULT_LABEL_COLUMN: Final[str] = "label"
DEFAULT_TRAIN_PATH: Final[str] = "data/raw/train-00000-of-00001.parquet"
DEFAULT_TEST_PATH: Final[str] = "data/raw/test-00000-of-00001.parquet"
DEFAULT_UNSUPERVISED_PATH: Final[str] = "data/raw/unsupervised-00000-of-00001.parquet"
DEFAULT_BASELINE_CONFIG_PATH: Final[str] = "configs/baseline.yaml"


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalize a raw review string with conservative cleanup only."""
    normalized_text = str(text)
    normalized_text = normalized_text.replace("<br />", " ")
    normalized_text = normalized_text.replace("<br/>", " ")
    normalized_text = normalized_text.replace("<br>", " ")
    normalized_text = WHITESPACE_PATTERN.sub(" ", normalized_text)
    normalized_text = normalized_text.strip()
    if lowercase:
        normalized_text = normalized_text.lower()
    return normalized_text


def tokenize_text(text: str) -> list[str]:
    """Split normalized text into word-level tokens while preserving punctuation."""
    if not text:
        return []
    return TOKEN_PATTERN.findall(text)


def text_to_ids(text: str, vocab: Vocab, lowercase: bool = True) -> list[int]:
    """Normalize, tokenize, and convert raw text into vocabulary ids."""
    normalized_text = normalize_text(text, lowercase=lowercase)
    tokens = tokenize_text(normalized_text)
    return tokens_to_ids(tokens, vocab)


def load_text_column_from_parquet(
    parquet_path: str | Path,
    text_column: str = DEFAULT_TEXT_COLUMN,
) -> list[str]:
    """Load the text column from a parquet file as a list of strings."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    data_frame = pd.read_parquet(path)
    if text_column not in data_frame.columns:
        raise ValueError(f"Missing text column '{text_column}' in {path}")

    return data_frame[text_column].astype(str).tolist()


def load_text_and_label_columns_from_parquet(
    parquet_path: str | Path,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> tuple[list[str], list[int]]:
    """Load text and label columns together from a labeled parquet split."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    data_frame = pd.read_parquet(path)
    if text_column not in data_frame.columns:
        raise ValueError(f"Missing text column '{text_column}' in {path}")
    if label_column not in data_frame.columns:
        raise ValueError(f"Missing label column '{label_column}' in {path}")

    texts = data_frame[text_column].astype(str).tolist()
    labels = data_frame[label_column].astype(int).tolist()
    return texts, labels


def build_vocab_from_parquet(
    parquet_path: str | Path,
    text_column: str = DEFAULT_TEXT_COLUMN,
    lowercase: bool = True,
    min_freq: int = 1,
    max_size: int | None = None,
) -> Vocab:
    """Build a vocabulary directly from the text column of one parquet split."""
    texts = load_text_column_from_parquet(parquet_path=parquet_path, text_column=text_column)
    token_lists = [tokenize_text(normalize_text(text, lowercase=lowercase)) for text in texts]
    return build_vocab_from_token_lists(token_lists=token_lists, min_freq=min_freq, max_size=max_size)


def load_supervised_examples_from_parquet(
    parquet_path: str | Path,
    vocab: Vocab,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    lowercase: bool = True,
) -> list[LabeledExample]:
    """Load labeled examples with raw text, input ids, and labels."""
    texts, labels = load_text_and_label_columns_from_parquet(
        parquet_path=parquet_path,
        text_column=text_column,
        label_column=label_column,
    )

    examples: list[LabeledExample] = []
    for text, label in zip(texts, labels, strict=True):
        examples.append(
            {
                "text": text,
                "input_ids": text_to_ids(text=text, vocab=vocab, lowercase=lowercase),
                "label": label,
            }
        )
    return examples


def load_unsupervised_examples_from_parquet(
    parquet_path: str | Path,
    vocab: Vocab,
    text_column: str = DEFAULT_TEXT_COLUMN,
    lowercase: bool = True,
) -> list[UnlabeledExample]:
    """Load unlabeled examples with raw text and input ids."""
    texts = load_text_column_from_parquet(parquet_path=parquet_path, text_column=text_column)
    return [
        {"text": text, "input_ids": text_to_ids(text=text, vocab=vocab, lowercase=lowercase)}
        for text in texts
    ]


def run_preprocessing_pipeline(
    train_path: str | Path,
    test_path: str | Path,
    unsupervised_path: str | Path,
    vocab_output_path: str | Path = DEFAULT_VOCAB_OUTPUT_PATH,
    train_processed_path: str | Path = DEFAULT_TRAIN_PROCESSED_PATH,
    test_processed_path: str | Path = DEFAULT_TEST_PROCESSED_PATH,
    unsupervised_processed_path: str | Path = DEFAULT_UNSUPERVISED_PROCESSED_PATH,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    lowercase: bool = True,
    min_freq: int = 1,
    max_size: int | None = None,
) -> PreprocessingArtifacts:
    """Build the vocab, write processed artifacts, and return all processed data in memory."""
    vocab = build_vocab_from_parquet(
        parquet_path=train_path,
        text_column=text_column,
        lowercase=lowercase,
        min_freq=min_freq,
        max_size=max_size,
    )
    save_vocab(vocab, vocab_output_path)

    train_examples = load_supervised_examples_from_parquet(
        parquet_path=train_path,
        vocab=vocab,
        text_column=text_column,
        label_column=label_column,
        lowercase=lowercase,
    )
    test_examples = load_supervised_examples_from_parquet(
        parquet_path=test_path,
        vocab=vocab,
        text_column=text_column,
        label_column=label_column,
        lowercase=lowercase,
    )
    unsupervised_examples = load_unsupervised_examples_from_parquet(
        parquet_path=unsupervised_path,
        vocab=vocab,
        text_column=text_column,
        lowercase=lowercase,
    )

    save_processed_supervised_examples(train_examples, train_processed_path)
    save_processed_supervised_examples(test_examples, test_processed_path)
    save_processed_unsupervised_examples(unsupervised_examples, unsupervised_processed_path)

    return {
        "vocab": vocab,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "unsupervised_examples": unsupervised_examples,
    }


def run_preprocessing_from_config(config: dict[str, object]) -> PreprocessingArtifacts:
    """Adapt a validated baseline-style config into the preprocessing pipeline."""
    data_config = config["data"]
    vocab_config = config["vocab"]

    if not isinstance(data_config, dict):
        raise TypeError("config['data'] must be a dictionary.")
    if not isinstance(vocab_config, dict):
        raise TypeError("config['vocab'] must be a dictionary.")

    return run_preprocessing_pipeline(
        train_path=DEFAULT_TRAIN_PATH,
        test_path=DEFAULT_TEST_PATH,
        unsupervised_path=DEFAULT_UNSUPERVISED_PATH,
        vocab_output_path=str(vocab_config.get("output_path", DEFAULT_VOCAB_OUTPUT_PATH)),
        train_processed_path=str(data_config.get("train_processed_path", DEFAULT_TRAIN_PROCESSED_PATH)),
        test_processed_path=str(data_config.get("test_processed_path", DEFAULT_TEST_PROCESSED_PATH)),
        unsupervised_processed_path=str(data_config.get("unsupervised_processed_path", DEFAULT_UNSUPERVISED_PROCESSED_PATH)),
        text_column=str(data_config["text_column"]),
        label_column=str(data_config["label_column"]),
        lowercase=bool(vocab_config["lowercase"]),
        min_freq=int(vocab_config["min_freq"]),
        max_size=int(vocab_config["max_size"]),
    )


def main(config_path: str = DEFAULT_BASELINE_CONFIG_PATH) -> None:
    """Run the local preprocessing pipeline from the baseline config."""
    config = load_and_validate_config(config_path, "baseline")
    artifacts = run_preprocessing_from_config(config)

    print("Preprocessing complete.")
    print(f"Vocab size: {len(artifacts['vocab'])}")
    print(f"Train examples: {len(artifacts['train_examples'])}")
    print(f"Test examples: {len(artifacts['test_examples'])}")
    print(f"Unsupervised examples: {len(artifacts['unsupervised_examples'])}")


if __name__ == "__main__":
    import sys

    selected_config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASELINE_CONFIG_PATH
    main(selected_config_path)
