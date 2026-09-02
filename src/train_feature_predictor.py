# src/train_feature_predictor.py

"""
Train the multi-task feature reconstruction network.

Pipeline
--------
Frozen DistilBERT embeddings
        +
original UCI sentiment targets
        +
original UCI LDA targets
        ↓
train / validation / test split
        ↓
target standardization using TRAIN ONLY
        ↓
shared multi-task FNN
        ↓
sentiment head + LDA head

Loss
----
    total_loss =
        sentiment_MSE
        +
        LDA_LOSS_WEIGHT * LDA_MSE

DistilBERT is NOT trained.
Only the FNN is trained.
"""

import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    BERT_EMBEDDINGS_PATH,
    BERT_EMBEDDING_METADATA_PATH,
    FEATURE_PREDICTOR_MODEL_PATH,
    FEATURE_TARGET_SCALER_PATH,
    FEATURE_SPLIT_INDICES_PATH,
    SENTIMENT_FEATURE_COLUMNS,
    LDA_FEATURE_COLUMNS,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    FNN_BATCH_SIZE,
    FNN_LEARNING_RATE,
    FNN_WEIGHT_DECAY,
    FNN_MAX_EPOCHS,
    FNN_EARLY_STOPPING_PATIENCE,
    FNN_MIN_DELTA,
    LDA_LOSS_WEIGHT,
    NORMALIZE_FEATURE_TARGETS,
)

from data_loader import (
    load_feature_prediction_dataset,
    get_reconstruction_targets,
)

from feature_predictor import (
    FeaturePredictor,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):
    """Set all relevant random seeds."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """Select the best available PyTorch device."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    embeddings = np.load(
        BERT_EMBEDDINGS_PATH
    )

    metadata = pd.read_csv(
        BERT_EMBEDDING_METADATA_PATH
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, "
            f"got {embeddings.shape}"
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding count does not match embedding metadata."
        )

    if "id" not in metadata.columns:
        raise ValueError(
            "Embedding metadata must contain `id`."
        )

    ids = metadata["id"].to_numpy()

    if len(np.unique(ids)) != len(ids):
        raise ValueError(
            "Embedding metadata contains duplicate IDs."
        )

    return embeddings.astype(np.float32), ids


# ============================================================
# ALIGN EMBEDDINGS WITH DATASET
# ============================================================

def align_embeddings(
    dataset,
    embeddings,
    embedding_ids,
):
    """
    Reorder embeddings so they exactly follow the canonical
    dataset ID ordering.
    """

    embedding_lookup = {
        int(article_id): index
        for index, article_id in enumerate(
            embedding_ids
        )
    }

    dataset_ids = dataset[
        "id"
    ].to_numpy()

    missing = [
        int(article_id)
        for article_id in dataset_ids
        if int(article_id)
        not in embedding_lookup
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} dataset articles have no "
            "corresponding BERT embedding."
        )

    indices = [
        embedding_lookup[
            int(article_id)
        ]
        for article_id in dataset_ids
    ]

    aligned = embeddings[
        indices
    ]

    return aligned


# ============================================================
# SPLIT DATA
# ============================================================

def create_splits(n_samples):

    all_indices = np.arange(
        n_samples
    )

    train_indices, temp_indices = (
        train_test_split(
            all_indices,
            test_size=(
                VALIDATION_RATIO
                + TEST_RATIO
            ),
            random_state=RANDOM_SEED,
            shuffle=True,
        )
    )

    relative_test_ratio = (
        TEST_RATIO
        / (
            VALIDATION_RATIO
            + TEST_RATIO
        )
    )

    validation_indices, test_indices = (
        train_test_split(
            temp_indices,
            test_size=relative_test_ratio,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
    )

    return (
        np.sort(train_indices),
        np.sort(validation_indices),
        np.sort(test_indices),
    )


# ============================================================
# TARGET SCALING
# ============================================================

def fit_target_scaler(
    sentiment_targets,
    lda_targets,
    train_indices,
):
    """
    Fit one scaler over the complete 21-dimensional target
    vector using TRAIN ONLY.
    """

    combined = np.concatenate(
        [
            sentiment_targets,
            lda_targets,
        ],
        axis=1,
    )

    scaler = StandardScaler()

    scaler.fit(
        combined[
            train_indices
        ]
    )

    return scaler


def transform_targets(
    sentiment_targets,
    lda_targets,
    scaler,
):
    """Transform 21-dimensional target matrix."""

    combined = np.concatenate(
        [
            sentiment_targets,
            lda_targets,
        ],
        axis=1,
    )

    transformed = scaler.transform(
        combined
    ).astype(np.float32)

    sentiment_dim = (
        sentiment_targets.shape[1]
    )

    sentiment_scaled = transformed[
        :,
        :sentiment_dim,
    ]

    lda_scaled = transformed[
        :,
        sentiment_dim:,
    ]

    return (
        sentiment_scaled,
        lda_scaled,
    )


# ============================================================
# BATCH ITERATION
# ============================================================

def iterate_batches(
    embeddings,
    sentiment_targets,
    lda_targets,
    indices,
    batch_size,
    shuffle,
):
    """Yield mini-batches."""

    indices = np.array(
        indices,
        dtype=np.int64,
    )

    if shuffle:
        np.random.shuffle(
            indices
        )

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        batch_indices = indices[
            start:start + batch_size
        ]

        yield (
            torch.from_numpy(
                embeddings[
                    batch_indices
                ]
            ),
            torch.from_numpy(
                sentiment_targets[
                    batch_indices
                ]
            ),
            torch.from_numpy(
                lda_targets[
                    batch_indices
                ]
            ),
        )


# ============================================================
# ONE EPOCH
# ============================================================

def run_epoch(
    model,
    embeddings,
    sentiment_targets,
    lda_targets,
    indices,
    optimizer,
    criterion,
    device,
    training,
):
    """
    Run one training or validation epoch.
    """

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_sentiment_loss = 0.0
    total_lda_loss = 0.0
    total_samples = 0

    for (
        batch_x,
        batch_sentiment,
        batch_lda,
    ) in iterate_batches(
        embeddings=embeddings,
        sentiment_targets=sentiment_targets,
        lda_targets=lda_targets,
        indices=indices,
        batch_size=FNN_BATCH_SIZE,
        shuffle=training,
    ):

        batch_x = batch_x.to(device)

        batch_sentiment = (
            batch_sentiment.to(device)
        )

        batch_lda = (
            batch_lda.to(device)
        )

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(
            training
        ):

            predicted_sentiment, predicted_lda = (
                model(batch_x)
            )

            sentiment_loss = criterion(
                predicted_sentiment,
                batch_sentiment,
            )

            lda_loss = criterion(
                predicted_lda,
                batch_lda,
            )

            total_batch_loss = (
                sentiment_loss
                + LDA_LOSS_WEIGHT * lda_loss
            )

            if training:

                total_batch_loss.backward()

                optimizer.step()

        batch_size_actual = (
            batch_x.shape[0]
        )

        total_loss += (
            total_batch_loss.item()
            * batch_size_actual
        )

        total_sentiment_loss += (
            sentiment_loss.item()
            * batch_size_actual
        )

        total_lda_loss += (
            lda_loss.item()
            * batch_size_actual
        )

        total_samples += batch_size_actual

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "sentiment_loss": (
            total_sentiment_loss
            / total_samples
        ),
        "lda_loss": (
            total_lda_loss
            / total_samples
        ),
    }


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(
    train_indices,
    validation_indices,
    test_indices,
):
    np.savez(
        FEATURE_SPLIT_INDICES_PATH,
        train=train_indices,
        validation=validation_indices,
        test=test_indices,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    device = get_device()

    print("=" * 90)
    print("MULTI-TASK FEATURE PREDICTOR TRAINING")
    print("=" * 90)

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load canonical data
    # --------------------------------------------------------

    dataset = load_feature_prediction_dataset()

    print(
        f"Articles in canonical dataset: "
        f"{len(dataset):,}"
    )

    sentiment_targets, lda_targets = (
        get_reconstruction_targets(
            dataset
        )
    )

    print(
        f"Sentiment targets: "
        f"{sentiment_targets.shape}"
    )

    print(
        f"LDA targets      : "
        f"{lda_targets.shape}"
    )

    # --------------------------------------------------------
    # Load BERT embeddings
    # --------------------------------------------------------

    embeddings, embedding_ids = (
        load_embeddings()
    )

    embeddings = align_embeddings(
        dataset,
        embeddings,
        embedding_ids,
    )

    print(
        f"BERT embeddings: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    (
        train_indices,
        validation_indices,
        test_indices,
    ) = create_splits(
        len(dataset)
    )

    save_splits(
        train_indices,
        validation_indices,
        test_indices,
    )

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
    # Scale targets
    # --------------------------------------------------------

    if NORMALIZE_FEATURE_TARGETS:

        scaler = fit_target_scaler(
            sentiment_targets,
            lda_targets,
            train_indices,
        )

        (
            sentiment_scaled,
            lda_scaled,
        ) = transform_targets(
            sentiment_targets,
            lda_targets,
            scaler,
        )

        with open(
            FEATURE_TARGET_SCALER_PATH,
            "wb",
        ) as handle:

            pickle.dump(
                scaler,
                handle,
            )

    else:

        scaler = None

        sentiment_scaled = (
            sentiment_targets
            .astype(np.float32)
        )

        lda_scaled = (
            lda_targets
            .astype(np.float32)
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FeaturePredictor().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FNN_LEARNING_RATE,
        weight_decay=FNN_WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    print(
        f"\nTrainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_state_dict = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        FNN_MAX_EPOCHS + 1,
    ):

        train_metrics = run_epoch(
            model=model,
            embeddings=embeddings,
            sentiment_targets=sentiment_scaled,
            lda_targets=lda_scaled,
            indices=train_indices,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            training=True,
        )

        validation_metrics = run_epoch(
            model=model,
            embeddings=embeddings,
            sentiment_targets=sentiment_scaled,
            lda_targets=lda_scaled,
            indices=validation_indices,
            optimizer=None,
            criterion=criterion,
            device=device,
            training=False,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics[
                "loss"
            ],
            "train_sentiment_loss": train_metrics[
                "sentiment_loss"
            ],
            "train_lda_loss": train_metrics[
                "lda_loss"
            ],
            "validation_loss": validation_metrics[
                "loss"
            ],
            "validation_sentiment_loss": validation_metrics[
                "sentiment_loss"
            ],
            "validation_lda_loss": validation_metrics[
                "lda_loss"
            ],
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_metrics['loss']:.6f} | "
            f"Val {validation_metrics['loss']:.6f} | "
            f"Val Sent {validation_metrics['sentiment_loss']:.6f} | "
            f"Val LDA {validation_metrics['lda_loss']:.6f}"
        )

        current_validation_loss = (
            validation_metrics[
                "loss"
            ]
        )

        if (
            best_validation_loss
            - current_validation_loss
            > FNN_MIN_DELTA
        ):

            best_validation_loss = (
                current_validation_loss
            )

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= FNN_EARLY_STOPPING_PATIENCE
        ):

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state_dict is None:
        raise RuntimeError(
            "No valid model checkpoint was produced."
        )

    model.load_state_dict(
        best_state_dict
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": 768,
            "sentiment_output_dim": 16,
            "lda_output_dim": 5,
            "best_validation_loss": best_validation_loss,
        },
        FEATURE_PREDICTOR_MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        FEATURE_PREDICTOR_MODEL_PATH.parent
        / "feature_predictor_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    final_validation_metrics = run_epoch(
        model=model,
        embeddings=embeddings,
        sentiment_targets=sentiment_scaled,
        lda_targets=lda_scaled,
        indices=validation_indices,
        optimizer=None,
        criterion=criterion,
        device=device,
        training=False,
    )

    print()
    print("=" * 90)
    print("TRAINING COMPLETE")
    print("=" * 90)

    print(
        f"Best validation loss : "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Final validation loss: "
        f"{final_validation_metrics['loss']:.6f}"
    )

    print(
        f"\nModel saved to:\n"
        f"{FEATURE_PREDICTOR_MODEL_PATH}"
    )

    if NORMALIZE_FEATURE_TARGETS:

        print(
            f"\nTarget scaler saved to:\n"
            f"{FEATURE_TARGET_SCALER_PATH}"
        )

    print(
        f"\nSplit indices saved to:\n"
        f"{FEATURE_SPLIT_INDICES_PATH}"
    )

    print(
        "\nMulti-task feature predictor training: PASSED"
    )


if __name__ == "__main__":
    main()