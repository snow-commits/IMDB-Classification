from __future__ import annotations

import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    """BiLSTM sentiment classifier with masked max pooling and a two-layer MLP head."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        classifier_hidden_dim: int,
        embedding_dropout: float,
        classifier_dropout: float,
    ) -> None:
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.linear1 = nn.Linear(self.output_dim, classifier_hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(classifier_hidden_dim, 2)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids to embeddings and apply embedding dropout."""
        embeddings = self.embedding(input_ids)
        return self.embedding_dropout(embeddings)

    # Unused helper; forward() uses forward_from_embeddings() directly.
    #     def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
    #         """Encode token ids into contextual BiLSTM representations."""
    #         embeddings = self.embed(input_ids)
    #         lstm_output, _ = self.bilstm(embeddings)
    #         return lstm_output

    def masked_max_pool(self, lstm_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply max pooling across time while ignoring padding positions."""
        expanded_mask = mask.unsqueeze(-1)
        masked_output = lstm_output.masked_fill(~expanded_mask, float("-inf"))
        return masked_output.max(dim=1).values

    def classify(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Apply the classifier head to pooled sequence representations."""
        hidden_output = self.classifier_dropout(pooled_output)
        hidden_output = self.linear1(hidden_output)
        hidden_output = self.relu(hidden_output)
        return self.linear2(hidden_output)

    def forward_from_embeddings(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the classifier from precomputed embeddings for VAT-style extensions."""
        lstm_output, _ = self.bilstm(embeddings)
        pooled_output = self.masked_max_pool(lstm_output, mask)
        return self.classify(pooled_output)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute binary sentiment logits from token ids and a padding mask."""
        embeddings = self.embed(input_ids)
        return self.forward_from_embeddings(embeddings, mask)
