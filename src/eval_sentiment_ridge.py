"""
Feature-wise evaluation of the Ridge sentiment predictor.

Loads the saved Ridge model + split indices (no retraining), recomputes
per-feature test metrics, and — if available — compares them side by
side against the FNN's per-feature metrics (SENTIMENT_METRICS_PATH) to
show which of the 16 sentiment features are recoverable and whether
Ridge or the FNN is the better predictor for each.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import config
from data_loader import load_feature_prediction_dataset, get_sentiment_targets

RIDGE_MODEL_PATH = config.MODELS_DIR / "sentiment_ridge_predictor.pkl"
RIDGE_EVAL_METRICS_PATH = config.OUTPUTS_DIR / "sentiment_ridge_eval_metrics.csv"
RIDGE_VS_FNN_PATH = config.OUTPUTS_DIR / "sentiment_ridge_vs_fnn.csv"


def load_ridge_artifacts():
    path = Path(RIDGE_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Ridge model not found:\n{path}\nRun train_sentiment_ridge.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_split(n_samples: int):
    path = Path(config.SENTIMENT_SPLIT_INDICES_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Split indices not found:\n{path}\n"
            "Run train_sentiment_predictor.py or train_sentiment_ridge.py first."
        )
    data = np.load(path)
    return data["train_indices"], data["validation_indices"], data["test_indices"]


def per_feature_metrics(y_test: np.ndarray, predictions: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, feature_name in enumerate(config.SENTIMENT_FEATURE_COLUMNS):
        y_true = y_test[:, i]
        y_pred = predictions[:, i]
        y_base = baseline[:, i]

        mse = mean_squared_error(y_true, y_pred)
        baseline_mse = mean_squared_error(y_true, y_base)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = np.nan if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot

        if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
            correlation = np.nan
        else:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]

        improvement = (
            (baseline_mse - mse) / baseline_mse * 100 if baseline_mse > 1e-12 else np.nan
        )

        rows.append({
            "feature": feature_name,
            "ridge_mse": mse,
            "baseline_mse": baseline_mse,
            "ridge_improvement_percent": improvement,
            "ridge_mae": mae,
            "ridge_rmse": rmse,
            "ridge_r2": r2,
            "ridge_correlation": correlation,
        })
    return pd.DataFrame(rows)


def run():
    artifacts = load_ridge_artifacts()
    model = artifacts["model"]
    train_mean = artifacts["train_mean"]
    train_std = artifacts["train_std"]

    dataset = load_feature_prediction_dataset()
    targets = get_sentiment_targets(dataset)
    embeddings = np.load(config.SENTIMENT_BERT_EMBEDDINGS_PATH)

    _, _, test_idx = load_split(len(targets))
    X_test, y_test = embeddings[test_idx], targets[test_idx]

    predictions_scaled = model.predict(X_test)
    predictions = predictions_scaled * train_std + train_mean
    baseline = np.tile(train_mean, (len(y_test), 1))

    metrics_df = per_feature_metrics(y_test, predictions, baseline)

    RIDGE_EVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(RIDGE_EVAL_METRICS_PATH, index=False)

    print("=" * 70)
    print("RIDGE — PER-FEATURE TEST METRICS")
    print("=" * 70)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    overall_mse = mean_squared_error(y_test, predictions)
    overall_baseline_mse = mean_squared_error(y_test, baseline)
    overall_improvement = (
        (overall_baseline_mse - overall_mse) / overall_baseline_mse * 100
        if overall_baseline_mse > 1e-12 else np.nan
    )
    print(f"\nOverall Ridge MSE: {overall_mse:.6f}")
    print(f"Overall baseline MSE: {overall_baseline_mse:.6f}")
    print(f"Overall improvement: {overall_improvement:.2f}%")

    # --------------------------------------------------------
    # Compare against FNN per-feature metrics if available
    # --------------------------------------------------------
    fnn_path = Path(config.SENTIMENT_METRICS_PATH)
    if fnn_path.exists():
        fnn_df = pd.read_csv(fnn_path)[
            ["feature", "mse", "improvement_percent", "r2", "correlation"]
        ].rename(columns={
            "mse": "fnn_mse",
            "improvement_percent": "fnn_improvement_percent",
            "r2": "fnn_r2",
            "correlation": "fnn_correlation",
        })

        comparison = metrics_df.merge(fnn_df, on="feature", how="left")
        comparison["ridge_beats_fnn"] = comparison["ridge_mse"] < comparison["fnn_mse"]

        comparison.to_csv(RIDGE_VS_FNN_PATH, index=False)

        print("\n" + "=" * 70)
        print("RIDGE vs FNN — PER FEATURE")
        print("=" * 70)
        print(
            comparison[
                ["feature", "ridge_mse", "fnn_mse", "ridge_beats_fnn",
                 "ridge_improvement_percent", "fnn_improvement_percent"]
            ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
        )

        n_ridge_wins = int(comparison["ridge_beats_fnn"].sum())
        print(f"\nRidge beats FNN on {n_ridge_wins}/{len(comparison)} features.")
        print(f"Saved comparison: {RIDGE_VS_FNN_PATH}")
    else:
        print(f"\n(No FNN metrics found at {fnn_path} — skipping comparison. "
              "Run train_sentiment_predictor.py to enable it.)")

    print(f"\nSaved: {RIDGE_EVAL_METRICS_PATH}")


if __name__ == "__main__":
    run()