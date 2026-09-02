# src/feature_predictor.py

"""
Multi-task Feedforward Neural Network for reconstructing the
original NLP feature representation from frozen DistilBERT
embeddings.

Architecture
------------
Frozen DistilBERT
       ↓
     768
       ↓
     256
       ↓
     128
      / \
     /   \
    ↓     ↓
   16      5
sentiment LDA

The DistilBERT encoder itself is NOT part of this trainable model.
Its embeddings are generated separately and kept frozen.

Outputs
-------
Sentiment head:
    16 original UCI sentiment features

LDA head:
    5 original UCI LDA features
"""

import torch
import torch.nn as nn

from config import (
    BERT_EMBEDDING_DIM,
    FNN_SHARED_DIM_1,
    FNN_SHARED_DIM_2,
    FNN_SENTIMENT_OUTPUT_DIM,
    FNN_LDA_OUTPUT_DIM,
    FNN_DROPOUT,
)


class FeaturePredictor(nn.Module):
    """
    Shared multi-task FNN.

    Input:
        768-dimensional frozen DistilBERT embedding

    Outputs:
        sentiment predictions: 16
        LDA predictions:        5
    """

    def __init__(
        self,
        input_dim=BERT_EMBEDDING_DIM,
        shared_dim_1=FNN_SHARED_DIM_1,
        shared_dim_2=FNN_SHARED_DIM_2,
        sentiment_output_dim=FNN_SENTIMENT_OUTPUT_DIM,
        lda_output_dim=FNN_LDA_OUTPUT_DIM,
        dropout=FNN_DROPOUT,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Shared representation
        # ----------------------------------------------------

        self.shared = nn.Sequential(
            nn.Linear(
                input_dim,
                shared_dim_1,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                shared_dim_1,
                shared_dim_2,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----------------------------------------------------
        # Sentiment task head
        # ----------------------------------------------------

        self.sentiment_head = nn.Linear(
            shared_dim_2,
            sentiment_output_dim,
        )

        # ----------------------------------------------------
        # LDA task head
        # ----------------------------------------------------

        self.lda_head = nn.Linear(
            shared_dim_2,
            lda_output_dim,
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Tensor of shape
                (batch_size, 768)

        Returns:
            sentiment_output:
                (batch_size, 16)

            lda_output:
                (batch_size, 5)
        """

        shared_representation = self.shared(x)

        sentiment_output = (
            self.sentiment_head(
                shared_representation
            )
        )

        lda_output = (
            self.lda_head(
                shared_representation
            )
        )

        return (
            sentiment_output,
            lda_output,
        )


def count_parameters(model):
    """Count trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():

    model = FeaturePredictor()

    print("=" * 80)
    print("MULTI-TASK FEATURE PREDICTOR")
    print("=" * 80)

    print(model)

    print(
        f"\nInput dimension       : "
        f"{BERT_EMBEDDING_DIM}"
    )

    print(
        f"Sentiment outputs    : "
        f"{FNN_SENTIMENT_OUTPUT_DIM}"
    )

    print(
        f"LDA outputs          : "
        f"{FNN_LDA_OUTPUT_DIM}"
    )

    print(
        f"Trainable parameters : "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Dummy forward pass
    # --------------------------------------------------------

    dummy_input = torch.randn(
        8,
        BERT_EMBEDDING_DIM,
    )

    model.eval()

    with torch.no_grad():

        sentiment_output, lda_output = (
            model(dummy_input)
        )

    print(
        f"\nInput shape          : "
        f"{dummy_input.shape}"
    )

    print(
        f"Sentiment shape      : "
        f"{sentiment_output.shape}"
    )

    print(
        f"LDA shape            : "
        f"{lda_output.shape}"
    )

    assert sentiment_output.shape == (
        8,
        FNN_SENTIMENT_OUTPUT_DIM,
    )

    assert lda_output.shape == (
        8,
        FNN_LDA_OUTPUT_DIM,
    )

    print(
        "\nArchitecture check: PASSED"
    )


if __name__ == "__main__":
    main()