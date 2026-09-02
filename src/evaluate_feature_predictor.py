# src/evaluate_feature_predictor.py

"""
Evaluate the trained multi-task feature reconstruction model.

Outputs
-------
For each of the 21 reconstructed features:

    MAE
    RMSE
    R2
    Pearson correlation

The model predicts:

    16 sentiment features
    5 LDA features

All evaluation is performed on the untouched TEST split.

Predictions are inverse-transformed back to their original UCI
feature scales before metrics are calculated.
"""

import pickle

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config import (
    BERT_EMBEDDINGS_PATH,
    BERT_EMBEDDING_METADATA_PATH,
    FEATURE_PREDICTOR_MODEL_PATH,
    FEATURE_TARGET_SCALER_PATH,
    FEATURE_SPLIT_INDICES_PATH,
    FEATURE_PREDICTIONS_PATH,
    FEATURE_PREDICTION_METRICS_PATH,
    SENTIMENT_FEATURE_COLUMNS,
    LDA_FEATURE_COLUMNS,
    RANDOM_SEED,
)

from data_loader import (
    load_feature_prediction_dataset,
    get_reconstruction_targets,
)

from feature_predictor import (
    FeaturePredictor,
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_and_align_embeddings(
    dataset,
):
    embeddings = np.load(
        BERT_EMBEDDINGS_PATH
    )

    metadata = pd.read_csv(
        BERT_EMBEDDING_METADATA_PATH
    )

    if "id" not in metadata.columns:
        raise ValueError(
            "Embedding metadata is missing `id`."
        )

    lookup = {
        int(article_id): index
        for index, article_id
        in enumerate(
            metadata["id"].to_numpy()
        )
    }

    indices = []

    for article_id in dataset[
        "id"
    ].to_numpy():

        article_id = int(
            article_id
        )

        if article_id not in lookup:
            raise ValueError(
                f"No BERT embedding found for "
                f"article ID {article_id}."
            )

        indices.append(
            lookup[article_id]
        )

    return embeddings[
        indices
    ].astype(np.float32)


# ============================================================
# METRICS
# ============================================================

def calculate_correlation(
    actual,
    predicted,
):
    """Calculate Pearson correlation."""

    if len(actual) < 2:
        return np.nan

    if np.std(actual) == 0.0:
        return np.nan

    if np.std(predicted) == 0.0:
        return np.nan

    return float(
        np.corrcoef(
            actual,
            predicted,
        )[0, 1]
    )


def calculate_metrics(
    actual,
    predicted,
):
    """Calculate regression metrics."""

    mse = mean_squared_error(
        actual,
        predicted,
    )

    return {
        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "rmse": float(
            np.sqrt(mse)
        ),
        "r2": float(
            r2_score(
                actual,
                predicted,
            )
        ),
        "correlation": calculate_correlation(
            actual,
            predicted,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("MULTI-TASK FEATURE PREDICTOR EVALUATION")
    print("=" * 90)

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load canonical dataset
    # --------------------------------------------------------

    dataset = load_feature_prediction_dataset()

    (
        sentiment_targets,
        lda_targets,
    ) = get_reconstruction_targets(
        dataset
    )

    # --------------------------------------------------------
    # Load BERT embeddings
    # --------------------------------------------------------

    embeddings = (
        load_and_align_embeddings(
            dataset
        )
    )

    # --------------------------------------------------------
    # Load split
    # --------------------------------------------------------

    split_data = np.load(
        FEATURE_SPLIT_INDICES_PATH
    )

    train_indices = split_data[
        "train"
    ]

    validation_indices = split_data[
        "validation"
    ]

    test_indices = split_data[
        "test"
    ]

    print(
        f"\nTrain      : "
        f"{len(train_indices):,}"
    )

    print(
        f"Validation : "
        f"{len(validation_indices):,}"
    )

    print(
        f"Test       : "
        f"{len(test_indices):,}"
    )

    # --------------------------------------------------------
    # Load target scaler
    # --------------------------------------------------------

    with open(
        FEATURE_TARGET_SCALER_PATH,
        "rb",
    ) as handle:

        scaler = pickle.load(
            handle
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    checkpoint = torch.load(
        FEATURE_PREDICTOR_MODEL_PATH,
        map_location=device,
    )

    model = FeaturePredictor().to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    test_embeddings = torch.from_numpy(
        embeddings[
            test_indices
        ]
    ).to(device)

    with torch.no_grad():

        predicted_sentiment, predicted_lda = (
            model(
                test_embeddings
            )
        )

    predicted_sentiment = (
        predicted_sentiment
        .cpu()
        .numpy()
    )

    predicted_lda = (
        predicted_lda
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Inverse-transform predictions
    # --------------------------------------------------------

    predicted_combined_scaled = (
        np.concatenate(
            [
                predicted_sentiment,
                predicted_lda,
            ],
            axis=1,
        )
    )

    predicted_combined = (
        scaler.inverse_transform(
            predicted_combined_scaled
        )
    )

    sentiment_dim = len(
        SENTIMENT_FEATURE_COLUMNS
    )

    predicted_sentiment = (
        predicted_combined[
            :,
            :sentiment_dim,
        ]
    )

    predicted_lda = (
        predicted_combined[
            :,
            sentiment_dim:,
        ]
    )

    actual_sentiment = (
        sentiment_targets[
            test_indices
        ]
    )

    actual_lda = (
        lda_targets[
            test_indices
        ]
    )

    # --------------------------------------------------------
    # Feature-level metrics
    # --------------------------------------------------------

    rows = []

    # Sentiment
    for index, feature in enumerate(
        SENTIMENT_FEATURE_COLUMNS
    ):

        metrics = calculate_metrics(
            actual_sentiment[:, index],
            predicted_sentiment[:, index],
        )

        rows.append({
            "group": "sentiment",
            "feature": feature,
            **metrics,
        })

    # LDA
    for index, feature in enumerate(
        LDA_FEATURE_COLUMNS
    ):

        metrics = calculate_metrics(
            actual_lda[:, index],
            predicted_lda[:, index],
        )

        rows.append({
            "group": "lda",
            "feature": feature,
            **metrics,
        })

    metrics_df = pd.DataFrame(
        rows
    )

    metrics_df.to_csv(
        FEATURE_PREDICTION_METRICS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_output = pd.DataFrame({
        "id": dataset.iloc[
            test_indices
        ]["id"].to_numpy(),
    })

    for index, feature in enumerate(
        SENTIMENT_FEATURE_COLUMNS
    ):

        prediction_output[
            f"actual_{feature}"
        ] = actual_sentiment[
            :,
            index,
        ]

        prediction_output[
            f"predicted_{feature}"
        ] = predicted_sentiment[
            :,
            index,
        ]

    for index, feature in enumerate(
        LDA_FEATURE_COLUMNS
    ):

        prediction_output[
            f"actual_{feature}"
        ] = actual_lda[
            :,
            index,
        ]

        prediction_output[
            f"predicted_{feature}"
        ] = predicted_lda[
            :,
            index,
        ]

    prediction_output.to_csv(
        FEATURE_PREDICTIONS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("FEATURE RECONSTRUCTION RESULTS")
    print("=" * 100)

    print(
        f"{'Group':<12}"
        f"{'Feature':<42}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'R2':>12}"
        f"{'Corr':>12}"
    )

    print("-" * 100)

    for _, row in metrics_df.iterrows():

        correlation = row[
            "correlation"
        ]

        correlation_text = (
            f"{correlation:.6f}"
            if np.isfinite(
                correlation
            )
            else "NaN"
        )

        print(
            f"{row['group']:<12}"
            f"{row['feature']:<42}"
            f"{row['mae']:>12.6f}"
            f"{row['rmse']:>12.6f}"
            f"{row['r2']:>12.6f}"
            f"{correlation_text:>12}"
        )

    print("=" * 100)

    # --------------------------------------------------------
    # Group summaries
    # --------------------------------------------------------

    print(
        "\nGROUP SUMMARIES"
    )

    for group in [
        "sentiment",
        "lda",
    ]:

        group_df = metrics_df[
            metrics_df["group"] == group
        ]

        print(
            f"\n{group.upper()}"
        )

        print(
            f"Mean MAE        : "
            f"{group_df['mae'].mean():.6f}"
        )

        print(
            f"Mean RMSE       : "
            f"{group_df['rmse'].mean():.6f}"
        )

        print(
            f"Mean R2         : "
            f"{group_df['r2'].mean():.6f}"
        )

        print(
            f"Mean Correlation: "
            f"{group_df['correlation'].mean():.6f}"
        )

    print(
        f"\nMetrics saved to:\n"
        f"{FEATURE_PREDICTION_METRICS_PATH}"
    )

    print(
        f"\nPredictions saved to:\n"
        f"{FEATURE_PREDICTIONS_PATH}"
    )

    print(
        "\nFeature predictor evaluation: PASSED"
    )


if __name__ == "__main__":
    main()