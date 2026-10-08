from __future__ import annotations

from typing import TypedDict

from torch.utils.data import Dataset


class LabeledExample(TypedDict):
    """A labeled sentiment example with raw text, token ids, and a binary class label."""

    text: str
    input_ids: list[int]
    label: int


class UnlabeledExample(TypedDict):
    """An unlabeled text example containing raw text and token ids."""

    text: str
    input_ids: list[int]


class LabeledTextDataset(Dataset):
    """PyTorch dataset wrapper for labeled sentiment examples."""

    def __init__(self, examples: list[LabeledExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        """Return the number of labeled examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> LabeledExample:
        """Return one labeled example by index."""
        return self.examples[index]


# Unused Dataset wrapper; current BiLSTM pipeline does not load unlabeled batches.
# class UnlabeledTextDataset(Dataset):
#     """PyTorch dataset wrapper for unlabeled text examples."""
#
#     def __init__(self, examples: list[UnlabeledExample]) -> None:
#         self.examples = examples
#
#     def __len__(self) -> int:
#         """Return the number of unlabeled examples."""
#         return len(self.examples)
#
#     def __getitem__(self, index: int) -> UnlabeledExample:
#         """Return one unlabeled example by index."""
#         return self.examples[index]
