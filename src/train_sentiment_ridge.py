"""
Ridge regression probe on frozen sentiment-BERT embeddings.
Diagnostic: is the MLP overfitting, or does the embedding lack signal
for the 16 Pattern-derived sentiment targets?

Mirrors train_sentiment_predictor.py exactly:
  - same split logic (reuses SENTIMENT_SPLIT_INDICES_PATH if present,
    else regenerates with identical rng calls and saves it)
  - same train-mean/std target standardization
  - same mean-baseline definition
Only the model differs: Ridge instead of the FNN, alpha chosen on the
validation set (not CV), matching the MLP's train/val/test protocol.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import config
from data_loader import load_feature_prediction_dataset, get_sentiment_targets

ALPHAS = [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0]

RIDGE_MODEL_PATH = config.MODELS_DIR / "sentiment_ridge_predictor.pkl"
RIDGE_PREDICTIONS_PATH = config.OUTPUTS_DIR / "sentiment_ridge_predictions.csv"
RIDGE_METRICS_PATH = config.OUTPUTS_DIR / "sentiment_ridge_metrics.csv"


def get_split_indices(n_samples: int):
    """Reuse the MLP's saved split if present; else regenerate identically."""

    path = Path(config.SENTIMENT_SPLIT_INDICES_PATH)

    if path.exists():
        data = np.load(path)
        return data["train_indices"], data["validation_indices"], data["test_indices"]

    indices = np.arange(n_samples)
    rng = np.random.default_rng(config.RANDOM_SEED)
    rng.shuffle(indices)

    test_count = int(round(n_samples * config.TEST_SIZE))
    validation_count = int(round(n_samples * config.VALIDATION_SIZE))

    test_indices = indices[:test_count]
    validation_indices = indices[test_count:test_count + validation_count]
    train_indices = indices[test_count + validation_count:]

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )

    return train_indices, validation_indices, test_indices


def run():
    dataset = load_feature_prediction_dataset()
    targets = get_sentiment_targets(dataset)

    embeddings_path = Path(config.SENTIMENT_BERT_EMBEDDINGS_PATH)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Sentiment-RoBERTa embeddings not found:\n{embeddings_path}\n"
            "Run first: python src/sentiment_bert_embeddings.py"
        )
    embeddings = np.load(embeddings_path)

    train_idx, val_idx, test_idx = get_split_indices(len(targets))

    X_train, X_val, X_test = embeddings[train_idx], embeddings[val_idx], embeddings[test_idx]
    y_train, y_val, y_test = targets[train_idx], targets[val_idx], targets[test_idx]

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    # --------------------------------------------------------
    # Target standardization (train-only mean/std, matches MLP)
    # --------------------------------------------------------
    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)

    y_train_s = (y_train - train_mean) / train_std
    y_val_s = (y_val - train_mean) / train_std

    # --------------------------------------------------------
    # Alpha selection on validation set (mirrors MLP's use of
    # val loss for model selection / early stopping)
    # --------------------------------------------------------
    best_alpha, best_val_mse, best_model = None, np.inf, None

    for alpha in ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train_s)
        val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val_s, val_pred)

        print(f"alpha={alpha:<8} val_mse={val_mse:.6f}")

        if val_mse < best_val_mse:
            best_alpha, best_val_mse, best_model = alpha, val_mse, model

    print(f"\nBest alpha: {best_alpha}  (val_mse={best_val_mse:.6f})")

    # --------------------------------------------------------
    # Test predictions (inverse standardization, matches MLP)
    # --------------------------------------------------------
    predictions_scaled = best_model.predict(X_test)
    predictions = predictions_scaled * train_std + train_mean

    baseline_predictions = np.tile(train_mean, (len(y_test), 1))

    model_mse = mean_squared_error(y_test, predictions)
    baseline_mse = mean_squared_error(y_test, baseline_predictions)
    improvement = (baseline_mse - model_mse) / baseline_mse * 100 if baseline_mse > 1e-12 else np.nan

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Ridge Test MSE:    {model_mse:.6f}")
    print(f"Mean Baseline MSE: {baseline_mse:.6f}")
    print(f"Improvement:       {improvement:.2f}%")

    # --------------------------------------------------------
    # Save model + scaler
    # --------------------------------------------------------
    RIDGE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RIDGE_MODEL_PATH, "wb") as f:
        pickle.dump(
            {"model": best_model, "alpha": best_alpha, "train_mean": train_mean, "train_std": train_std},
            f,
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------
    prediction_df = pd.DataFrame({"sample_index": test_idx})
    for i, feature_name in enumerate(config.SENTIMENT_FEATURE_COLUMNS):
        prediction_df[f"true_{feature_name}"] = y_test[:, i]
        prediction_df[f"pred_{feature_name}"] = predictions[:, i]

    RIDGE_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(RIDGE_PREDICTIONS_PATH, index=False)

    # --------------------------------------------------------
    # Per-feature metrics (same schema as SENTIMENT_METRICS_PATH)
    # --------------------------------------------------------
    metrics = []
    for i, feature_name in enumerate(config.SENTIMENT_FEATURE_COLUMNS):
        y_true = y_test[:, i]
        y_pred = predictions[:, i]
        y_base = baseline_predictions[:, i]

        mse = mean_squared_error(y_true, y_pred)
        baseline_feature_mse = mean_squared_error(y_true, y_base)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = np.nan if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot

        if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
            correlation = np.nan
        else:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]

        feature_improvement = (
            (baseline_feature_mse - mse) / baseline_feature_mse * 100
            if baseline_feature_mse > 1e-12 else np.nan
        )

        metrics.append({
            "feature": feature_name,
            "mse": mse,
            "baseline_mse": baseline_feature_mse,
            "improvement_percent": feature_improvement,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "correlation": correlation,
        })

    metrics_df = pd.DataFrame(metrics)
    RIDGE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(RIDGE_METRICS_PATH, index=False)

    print("\n" + "=" * 60)
    print("PER-FEATURE RESULTS")
    print("=" * 60)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSaved:")
    print(f"Model:       {RIDGE_MODEL_PATH}")
    print(f"Predictions: {RIDGE_PREDICTIONS_PATH}")
    print(f"Metrics:     {RIDGE_METRICS_PATH}")


if __name__ == "__main__":
    run()