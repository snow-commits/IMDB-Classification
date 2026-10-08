from __future__ import annotations

import torch

from src.data.dataset import LabeledExample
# from src.data.dataset import UnlabeledExample


def pad_sequences(
    sequences: list[list[int]],
    pad_value: int,
) -> list[list[int]]:
    """Pad variable-length token-id sequences to a common batch length."""
    if not sequences:
        return []

    max_length = max(len(sequence) for sequence in sequences)

    padded_sequences = []
    for sequence in sequences:
        padding = [pad_value] * (max_length - len(sequence))
        padded_sequences.append(sequence + padding)

    return padded_sequences


def get_sequence_lengths(sequences: list[list[int]]) -> list[int]:
    """Return the original lengths of each unpadded sequence."""
    return [len(sequence) for sequence in sequences]


def build_padding_mask(
    padded_sequences: list[list[int]],
    pad_value: int,
) -> list[list[int]]:
    """Build a mask with True-like positions for real tokens and False-like positions for padding."""
    mask = []

    for sequence in padded_sequences:
        row = [0 if token == pad_value else 1 for token in sequence]
        mask.append(row)

    return mask


def collate_labeled_batch(
    batch: list[LabeledExample],
    pad_value: int,
) -> dict[str, torch.Tensor]:
    """Collate labeled examples into padded tensors for classifier training."""
    input_id_sequences = [sample["input_ids"] for sample in batch]
    labels = [sample["label"] for sample in batch]

    lengths = get_sequence_lengths(input_id_sequences)
    padded_input_ids = pad_sequences(input_id_sequences, pad_value=pad_value)
    mask = build_padding_mask(padded_input_ids, pad_value=pad_value)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.bool),
    }


# Unused unlabeled collate path; current BiLSTM pipeline trains/evaluates labeled data only.
# def collate_unlabeled_batch(
#     batch: list[UnlabeledExample],
#     pad_value: int,
# ) -> dict[str, torch.Tensor]:
#     """Collate unlabeled examples into padded tensors for VAT or LM pretraining."""
#     input_id_sequences = [sample["input_ids"] for sample in batch]
#
#     lengths = get_sequence_lengths(input_id_sequences)
#     padded_input_ids = pad_sequences(input_id_sequences, pad_value=pad_value)
#     mask = build_padding_mask(padded_input_ids, pad_value=pad_value)
#
#     return {
#         "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
#         "lengths": torch.tensor(lengths, dtype=torch.long),
#         "mask": torch.tensor(mask, dtype=torch.bool),
#     }
