# src/sentiment_predictor.py

"""
Sentiment-only Feedforward Neural Network.

Purpose
-------
Predict the 16 original UCI sentiment features from frozen
DistilBERT mean-pooled article embeddings.

Input:
    768-dimensional frozen DistilBERT embedding

Output:
    16 original UCI sentiment features

Architecture:
    768 -> 256 -> 128 -> 16

DistilBERT is never fine-tuned.
Only this FNN is trained.
"""

import torch
import torch.nn as nn

from config import (
    BERT_EMBEDDING_DIM,
    FNN_SHARED_DIM_1,
    FNN_SHARED_DIM_2,
    SENTIMENT_FEATURE_COUNT,
    FNN_DROPOUT,
)


class SentimentPredictor(nn.Module):
    """
    Feedforward neural network for reconstructing the original
    UCI sentiment feature representation.
    """

    def __init__(
        self,
        input_dim=BERT_EMBEDDING_DIM,
        hidden_dim1=FNN_SHARED_DIM_1,
        hidden_dim2=FNN_SHARED_DIM_2,
        output_dim=SENTIMENT_FEATURE_COUNT,
        dropout=FNN_DROPOUT,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim1,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim1,
                hidden_dim2,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim2,
                output_dim,
            ),
        )

    def forward(self, x):
        """
        Args:
            x:
                Tensor of shape (batch_size, 768)

        Returns:
            Tensor of shape (batch_size, 16)
        """

        return self.network(x)


def count_parameters(model):
    """Return number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    """Run an architecture sanity check."""

    model = SentimentPredictor()

    print("=" * 80)
    print("SENTIMENT PREDICTOR")
    print("=" * 80)

    print(model)

    print(
        f"\nInput dimension   : "
        f"{BERT_EMBEDDING_DIM}"
    )

    print(
        f"Output dimension  : "
        f"{SENTIMENT_FEATURE_COUNT}"
    )

    print(
        f"Trainable params  : "
        f"{count_parameters(model):,}"
    )

    dummy_input = torch.randn(
        8,
        BERT_EMBEDDING_DIM,
    )

    model.eval()

    with torch.no_grad():
        dummy_output = model(
            dummy_input
        )

    print(
        f"\nInput shape  : "
        f"{dummy_input.shape}"
    )

    print(
        f"Output shape : "
        f"{dummy_output.shape}"
    )

    assert dummy_output.shape == (
        8,
        SENTIMENT_FEATURE_COUNT,
    )

    print(
        "\nArchitecture check: PASSED"
    )


if __name__ == "__main__":
    main()