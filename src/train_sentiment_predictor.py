"""
Train a sentiment-feature predictor using frozen,
sentiment-fine-tuned RoBERTa embeddings.

Pipeline:
    title + article text
        ↓
    frozen sentiment-fine-tuned RoBERTa
        ↓
    768-dimensional embedding
        ↓
    768 → 256 → 128 → 16 FNN
        ↓
    16 original UCI sentiment features

The Transformer is completely frozen.
Only the FNN prediction head is trained.
"""


from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

from config import (
    SENTIMENT_BERT_EMBEDDINGS_PATH,
    SENTIMENT_BERT_EMBEDDING_DIM,

    FNN_BATCH_SIZE,
    FNN_LEARNING_RATE,
    FNN_WEIGHT_DECAY,
    FNN_MAX_EPOCHS,
    FNN_PATIENCE,
    FNN_MIN_DELTA,

    RANDOM_SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
    TARGET_STANDARDIZATION,

    SENTIMENT_TARGET_SCALER_PATH,
    SENTIMENT_SPLIT_INDICES_PATH,
    SENTIMENT_MODEL_PATH,
    SENTIMENT_METRICS_PATH,
    SENTIMENT_PREDICTIONS_PATH,
    SENTIMENT_TRAINING_HISTORY_PATH,

    SENTIMENT_FEATURE_COLUMNS,
)

from data_loader import (
    load_feature_prediction_dataset,
    get_sentiment_targets,
)

from sentiment_predictor import SentimentPredictor


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD DATASET
# ============================================================

dataset = load_feature_prediction_dataset()

targets = get_sentiment_targets(dataset)


# ============================================================
# LOAD SENTIMENT-ROBERTA EMBEDDINGS
# ============================================================

embeddings_path = Path(
    SENTIMENT_BERT_EMBEDDINGS_PATH
)

if not embeddings_path.exists():
    raise FileNotFoundError(
        "Sentiment-RoBERTa embeddings were not found:\n"
        f"{embeddings_path}\n\n"
        "Run first:\n"
        "python src/sentiment_bert_embeddings.py"
    )

embeddings = np.load(
    embeddings_path
)


# ============================================================
# DATA VALIDATION
# ============================================================

if len(embeddings) != len(targets):
    raise ValueError(
        "Embedding/target count mismatch:\n"
        f"Embeddings: {len(embeddings)}\n"
        f"Targets:   {len(targets)}"
    )


if embeddings.ndim != 2:
    raise ValueError(
        f"Expected 2-D embeddings, got shape "
        f"{embeddings.shape}"
    )


if embeddings.shape[1] != SENTIMENT_BERT_EMBEDDING_DIM:
    raise ValueError(
        "Unexpected embedding dimension:\n"
        f"Expected: {SENTIMENT_BERT_EMBEDDING_DIM}\n"
        f"Actual:   {embeddings.shape[1]}"
    )


expected_target_dim = len(
    SENTIMENT_FEATURE_COLUMNS
)

if targets.shape[1] != expected_target_dim:
    raise ValueError(
        "Unexpected target dimension:\n"
        f"Expected: {expected_target_dim}\n"
        f"Actual:   {targets.shape[1]}"
    )


print(f"Number of articles: {len(targets)}")
print(f"Embedding shape:    {embeddings.shape}")
print(f"Target shape:       {targets.shape}")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n_samples = len(targets)

indices = np.arange(n_samples)

rng = np.random.default_rng(
    RANDOM_SEED
)

rng.shuffle(indices)


test_count = int(
    round(n_samples * TEST_SIZE)
)

validation_count = int(
    round(n_samples * VALIDATION_SIZE)
)


# Test
test_indices = indices[
    :test_count
]


# Validation
validation_start = test_count

validation_end = (
    test_count
    + validation_count
)

validation_indices = indices[
    validation_start:validation_end
]


# Train
train_indices = indices[
    validation_end:
]


print()
print("Split sizes:")
print(f"Train:      {len(train_indices)}")
print(f"Validation: {len(validation_indices)}")
print(f"Test:       {len(test_indices)}")


# ============================================================
# SAVE SPLIT INDICES
# ============================================================

SENTIMENT_SPLIT_INDICES_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez(
    SENTIMENT_SPLIT_INDICES_PATH,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
)


# ============================================================
# SPLIT DATA
# ============================================================

X_train = embeddings[
    train_indices
]

X_val = embeddings[
    validation_indices
]

X_test = embeddings[
    test_indices
]


y_train = targets[
    train_indices
]

y_val = targets[
    validation_indices
]

y_test = targets[
    test_indices
]


# ============================================================
# TARGET STANDARDIZATION
# ============================================================

if TARGET_STANDARDIZATION:

    train_mean = y_train.mean(
        axis=0
    )

    train_std = y_train.std(
        axis=0
    )

    # Protect against constant target features.
    train_std = np.where(
        train_std < 1e-8,
        1.0,
        train_std,
    )

    y_train_scaled = (
        y_train - train_mean
    ) / train_std

    y_val_scaled = (
        y_val - train_mean
    ) / train_std

    y_test_scaled = (
        y_test - train_mean
    ) / train_std

    SENTIMENT_TARGET_SCALER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaler_data = {
        "mean": train_mean,
        "std": train_std,
    }

    with open(
        SENTIMENT_TARGET_SCALER_PATH,
        "wb",
    ) as f:

        pickle.dump(
            scaler_data,
            f,
        )

else:

    y_train_scaled = y_train.copy()
    y_val_scaled = y_val.copy()
    y_test_scaled = y_test.copy()


# ============================================================
# TORCH DATASETS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32,
)

y_train_tensor = torch.tensor(
    y_train_scaled,
    dtype=torch.float32,
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32,
)

y_val_tensor = torch.tensor(
    y_val_scaled,
    dtype=torch.float32,
)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor,
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=FNN_BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=FNN_BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# MODEL
# ============================================================

model = SentimentPredictor(
    input_dim=SENTIMENT_BERT_EMBEDDING_DIM,
    output_dim=len(SENTIMENT_FEATURE_COLUMNS),
)

model = model.to(DEVICE)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=FNN_LEARNING_RATE,
    weight_decay=FNN_WEIGHT_DECAY,
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

best_state_dict = None

epochs_without_improvement = 0

history = []


print()
print("Starting training...")
print("=" * 60)


for epoch in range(
    1,
    FNN_MAX_EPOCHS + 1,
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0

    train_count = 0


    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(
            DEVICE
        )

        batch_y = batch_y.to(
            DEVICE
        )


        optimizer.zero_grad()

        predictions = model(
            batch_X
        )

        loss = criterion(
            predictions,
            batch_y,
        )

        loss.backward()

        optimizer.step()


        batch_size = batch_X.size(0)

        train_loss_sum += (
            loss.item()
            * batch_size
        )

        train_count += batch_size


    train_loss = (
        train_loss_sum
        / train_count
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0

    val_count = 0


    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )


            predictions = model(
                batch_X
            )

            loss = criterion(
                predictions,
                batch_y,
            )


            batch_size = batch_X.size(0)

            val_loss_sum += (
                loss.item()
                * batch_size
            )

            val_count += batch_size


    val_loss = (
        val_loss_sum
        / val_count
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history.append(
        {
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": val_loss,
        }
    )


    print(
        f"Epoch {epoch:03d} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MSE: {val_loss:.6f}"
    )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    improvement = (
        best_val_loss
        - val_loss
    )


    if improvement > FNN_MIN_DELTA:

        best_val_loss = val_loss

        best_state_dict = {
            key: value.detach()
            .cpu()
            .clone()
            for key, value
            in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= FNN_PATIENCE
        ):

            print()
            print(
                f"Early stopping at epoch "
                f"{epoch}."
            )

            break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state_dict is None:

    raise RuntimeError(
        "No valid model checkpoint was produced."
    )


model.load_state_dict(
    best_state_dict
)

model = model.to(
    DEVICE
)


# ============================================================
# SAVE MODEL
# ============================================================

SENTIMENT_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    model.state_dict(),
    SENTIMENT_MODEL_PATH,
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    SENTIMENT_TRAINING_HISTORY_PATH,
    index=False,
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

model.eval()


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
).to(DEVICE)


with torch.no_grad():

    predictions_scaled = (
        model(
            X_test_tensor
        )
        .cpu()
        .numpy()
    )


# ============================================================
# INVERSE STANDARDIZATION
# ============================================================

if TARGET_STANDARDIZATION:

    predictions = (
        predictions_scaled
        * train_std
        + train_mean
    )

else:

    predictions = (
        predictions_scaled
    )


# ============================================================
# MEAN BASELINE
# ============================================================

if TARGET_STANDARDIZATION:

    baseline_mean = train_mean

else:

    baseline_mean = y_train.mean(
        axis=0
    )


baseline_predictions = np.tile(
    baseline_mean,
    (
        len(y_test),
        1,
    ),
)


# ============================================================
# OVERALL METRICS
# ============================================================

model_mse = mean_squared_error(
    y_test,
    predictions,
)

baseline_mse = mean_squared_error(
    y_test,
    baseline_predictions,
)


if baseline_mse > 1e-12:

    improvement = (
        (
            baseline_mse
            - model_mse
        )
        / baseline_mse
        * 100
    )

else:

    improvement = np.nan


print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"FNN Test MSE:      "
    f"{model_mse:.6f}"
)

print(
    f"Mean Baseline MSE: "
    f"{baseline_mse:.6f}"
)

print(
    f"Improvement:       "
    f"{improvement:.2f}%"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "sample_index": test_indices,
    }
)


for i, feature_name in enumerate(
    SENTIMENT_FEATURE_COLUMNS
):

    prediction_df[
        f"true_{feature_name}"
    ] = y_test[:, i]

    prediction_df[
        f"pred_{feature_name}"
    ] = predictions[:, i]


prediction_df.to_csv(
    SENTIMENT_PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# PER-FEATURE METRICS
# ============================================================

metrics = []


for i, feature_name in enumerate(
    SENTIMENT_FEATURE_COLUMNS
):

    y_true = y_test[:, i]

    y_pred = predictions[:, i]

    y_base = baseline_predictions[:, i]


    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = mean_squared_error(
        y_true,
        y_pred,
    )


    baseline_feature_mse = (
        mean_squared_error(
            y_true,
            y_base,
        )
    )


    # --------------------------------------------------------
    # MAE / RMSE
    # --------------------------------------------------------

    mae = np.mean(
        np.abs(
            y_true - y_pred
        )
    )

    rmse = np.sqrt(
        mse
    )


    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (
            y_true
            - np.mean(y_true)
        ) ** 2
    )


    if ss_tot < 1e-12:

        r2 = np.nan

    else:

        r2 = (
            1.0
            - ss_res / ss_tot
        )


    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    if (
        np.std(y_true) < 1e-12
        or np.std(y_pred) < 1e-12
    ):

        correlation = np.nan

    else:

        correlation = np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]


    # --------------------------------------------------------
    # BASELINE IMPROVEMENT
    # --------------------------------------------------------

    if (
        baseline_feature_mse
        > 1e-12
    ):

        feature_improvement = (
            (
                baseline_feature_mse
                - mse
            )
            / baseline_feature_mse
            * 100
        )

    else:

        feature_improvement = np.nan


    metrics.append(
        {
            "feature": feature_name,
            "mse": mse,
            "baseline_mse": baseline_feature_mse,
            "improvement_percent":
                feature_improvement,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "correlation": correlation,
        }
    )


metrics_df = pd.DataFrame(
    metrics
)


# ============================================================
# SAVE METRICS
# ============================================================

SENTIMENT_METRICS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

metrics_df.to_csv(
    SENTIMENT_METRICS_PATH,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("PER-FEATURE RESULTS")
print("=" * 60)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
)


print()
print("Saved:")
print(
    f"Model:       "
    f"{SENTIMENT_MODEL_PATH}"
)

print(
    f"Scaler:      "
    f"{SENTIMENT_TARGET_SCALER_PATH}"
)

print(
    f"Splits:      "
    f"{SENTIMENT_SPLIT_INDICES_PATH}"
)

print(
    f"Predictions: "
    f"{SENTIMENT_PREDICTIONS_PATH}"
)

print(
    f"Metrics:     "
    f"{SENTIMENT_METRICS_PATH}"
)

print(
    f"History:     "
    f"{SENTIMENT_TRAINING_HISTORY_PATH}"
)