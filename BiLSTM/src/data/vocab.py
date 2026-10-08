from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PAD_TOKEN: Final[str] = "<pad>"
UNK_TOKEN: Final[str] = "<unk>"


@dataclass(frozen=True, slots=True)
class Vocab:
    """Vocabulary lookup tables for token/id conversion."""

    token_to_id: dict[str, int]
    id_to_token: list[str]

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self.id_to_token)

    def token_to_index(self, token: str) -> int:
        """Map a token to its integer id with unknown-token fallback."""
        return self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])

    # Unused reverse lookup helper; current pipeline only maps tokens to ids.
    #     def index_to_token(self, index: int) -> str:
    #         """Map an integer id back to its token string."""
    #         return self.id_to_token[index]


def build_vocab_from_token_lists(
    token_lists: list[list[str]],
    min_freq: int = 1,
    max_size: int | None = None,
) -> Vocab:
    """Build a vocabulary ordered by descending frequency then alphabetically."""
    counter: Counter[str] = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab_tokens = [PAD_TOKEN, UNK_TOKEN]
    sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    for token, frequency in sorted_tokens:
        if frequency < min_freq:
            continue
        if token in {PAD_TOKEN, UNK_TOKEN}:
            continue
        if max_size is not None and len(vocab_tokens) >= max_size:
            break
        vocab_tokens.append(token)

    token_to_id = {token: index for index, token in enumerate(vocab_tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=vocab_tokens)


def save_vocab(vocab: Vocab, output_path: str | Path) -> None:
    """Persist the vocabulary to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "token_to_id": vocab.token_to_id,
        "id_to_token": vocab.id_to_token,
    }
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)


def load_vocab(input_path: str | Path) -> Vocab:
    """Load a vocabulary from JSON."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Vocab file not found: {path}")

    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    token_to_id = payload["token_to_id"]
    id_to_token = payload["id_to_token"]
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)


def tokens_to_ids(tokens: list[str], vocab: Vocab) -> list[int]:
    """Convert a token list into vocabulary ids."""
    return [vocab.token_to_index(token) for token in tokens]
