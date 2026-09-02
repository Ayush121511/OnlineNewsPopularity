# src/text_feature_construction.py

"""
Diagnostic version of deterministic word-feature reconstruction.

Reconstructs:
    n_tokens_title
    n_tokens_content
    n_unique_tokens
    n_non_stop_words
    n_non_stop_unique_tokens
    average_token_length

Also prints article-level comparisons between:
    Original UCI feature values
    Reconstructed feature values

The purpose is to diagnose why the current reconstruction
correlations are unexpectedly low.
"""

import re
import numpy as np
import pandas as pd

from config import (
    DIRECT_TEXT_FEATURE_COLUMNS,
    WORD_FEATURE_VALIDATION_PATH,
)
from data_loader import load_feature_prediction_dataset


# ============================================================
# TOKENIZATION
# ============================================================

WORD_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?"
)


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all",
    "am", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "don",
    "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
}


def tokenize_words(text):
    """Tokenize text into lowercase word tokens."""

    if not isinstance(text, str):
        text = "" if text is None else str(text)

    return WORD_PATTERN.findall(text.lower())


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_word_features(title, text):
    """Calculate the six reconstructed word features."""

    title_tokens = tokenize_words(title)
    content_tokens = tokenize_words(text)

    n_tokens_title = len(title_tokens)
    n_tokens_content = len(content_tokens)

    if n_tokens_content == 0:
        return {
            "n_tokens_title": 0.0,
            "n_tokens_content": 0.0,
            "n_unique_tokens": 0.0,
            "n_non_stop_words": 0.0,
            "n_non_stop_unique_tokens": 0.0,
            "average_token_length": 0.0,
        }

    unique_tokens = set(content_tokens)

    non_stop_tokens = [
        token
        for token in content_tokens
        if token not in STOP_WORDS
    ]

    unique_non_stop_tokens = set(non_stop_tokens)

    return {
        "n_tokens_title": float(n_tokens_title),

        "n_tokens_content": float(
            n_tokens_content
        ),

        "n_unique_tokens": float(
            len(unique_tokens)
            / n_tokens_content
        ),

        "n_non_stop_words": float(
            len(non_stop_tokens)
            / n_tokens_content
        ),

        "n_non_stop_unique_tokens": float(
            len(unique_non_stop_tokens)
            / n_tokens_content
        ),

        "average_token_length": float(
            sum(len(token) for token in content_tokens)
            / n_tokens_content
        ),
    }


# ============================================================
# RECONSTRUCT
# ============================================================

def reconstruct_word_features(dataset):
    """Reconstruct all six word features."""

    rows = []

    for _, row in dataset.iterrows():

        features = calculate_word_features(
            row["title"],
            row["text"],
        )

        rows.append({
            "id": row["id"],
            **features,
        })

    return pd.DataFrame(rows)


# ============================================================
# METRICS
# ============================================================

def calculate_mae(original, reconstructed):
    return float(
        np.mean(
            np.abs(
                original - reconstructed
            )
        )
    )


def calculate_rmse(original, reconstructed):
    return float(
        np.sqrt(
            np.mean(
                (original - reconstructed) ** 2
            )
        )
    )


def calculate_correlation(original, reconstructed):
    if len(original) < 2:
        return np.nan

    if np.std(original) == 0:
        return np.nan

    if np.std(reconstructed) == 0:
        return np.nan

    return float(
        np.corrcoef(
            original,
            reconstructed
        )[0, 1]
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_word_features(dataset, reconstructed):

    merged = dataset[
        ["id"] + DIRECT_TEXT_FEATURE_COLUMNS
    ].merge(
        reconstructed,
        on="id",
        how="inner",
        suffixes=(
            "_original",
            "_reconstructed",
        ),
        validate="one_to_one",
    )

    results = []

    for feature in DIRECT_TEXT_FEATURE_COLUMNS:

        original = merged[
            f"{feature}_original"
        ].to_numpy(dtype=np.float64)

        reconstructed_values = merged[
            f"{feature}_reconstructed"
        ].to_numpy(dtype=np.float64)

        results.append({
            "feature": feature,
            "mae": calculate_mae(
                original,
                reconstructed_values,
            ),
            "rmse": calculate_rmse(
                original,
                reconstructed_values,
            ),
            "correlation": calculate_correlation(
                original,
                reconstructed_values,
            ),
        })

    return pd.DataFrame(results), merged


# ============================================================
# DIAGNOSTIC PRINT
# ============================================================

def print_article_diagnostic(
    dataset,
    merged,
    n_articles=10,
):
    """
    Print side-by-side diagnostics for a small set of articles.
    """

    print()
    print("=" * 100)
    print("ARTICLE-LEVEL DIAGNOSTIC")
    print("=" * 100)

    sample = dataset.head(n_articles)

    for _, row in sample.iterrows():

        article_id = row["id"]

        comparison = merged[
            merged["id"] == article_id
        ].iloc[0]

        print()
        print("-" * 100)

        print(f"ID: {article_id}")

        print()
        print("TITLE:")
        print(str(row["title"])[:500])

        print()
        print(
            "TEXT LENGTH (characters): "
            f"{len(str(row['text'])):,}"
        )

        print()
        print("TEXT PREVIEW:")
        text_preview = (
            str(row["text"])
            .replace("\n", " ")
            .replace("\r", " ")
        )

        print(text_preview[:500])

        print()
        print(
            f"{'Feature':<32}"
            f"{'Original UCI':>20}"
            f"{'Reconstructed':>20}"
            f"{'Difference':>20}"
        )

        print("-" * 92)

        for feature in DIRECT_TEXT_FEATURE_COLUMNS:

            original = comparison[
                f"{feature}_original"
            ]

            reconstructed = comparison[
                f"{feature}_reconstructed"
            ]

            difference = (
                reconstructed - original
            )

            print(
                f"{feature:<32}"
                f"{original:>20.6f}"
                f"{reconstructed:>20.6f}"
                f"{difference:>20.6f}"
            )

    print()
    print("=" * 100)


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print()
    print("=" * 100)
    print("WORD FEATURE RECONSTRUCTION SUMMARY")
    print("=" * 100)

    print(
        f"{'Feature':<32}"
        f"{'MAE':>15}"
        f"{'RMSE':>15}"
        f"{'Correlation':>20}"
    )

    print("-" * 100)

    for _, row in results.iterrows():

        correlation = row["correlation"]

        correlation_text = (
            f"{correlation:.6f}"
            if np.isfinite(correlation)
            else "NaN"
        )

        print(
            f"{row['feature']:<32}"
            f"{row['mae']:>15.6f}"
            f"{row['rmse']:>15.6f}"
            f"{correlation_text:>20}"
        )

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("TEXT FEATURE RECONSTRUCTION DIAGNOSTIC")
    print("=" * 100)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    dataset = load_feature_prediction_dataset()

    print(
        f"\nArticles loaded: {len(dataset):,}"
    )

    # --------------------------------------------------------
    # Reconstruct
    # --------------------------------------------------------

    reconstructed = reconstruct_word_features(
        dataset
    )

    print(
        "Features reconstructed: "
        f"{len(DIRECT_TEXT_FEATURE_COLUMNS)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    results, merged = validate_word_features(
        dataset,
        reconstructed,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results.to_csv(
        WORD_FEATURE_VALIDATION_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print_summary(results)

    # --------------------------------------------------------
    # Print article diagnostics
    # --------------------------------------------------------

    print_article_diagnostic(
        dataset,
        merged,
        n_articles=10,
    )

    print(
        f"\nValidation results saved to:\n"
        f"{WORD_FEATURE_VALIDATION_PATH}"
    )

    print(
        "\nDiagnostic run completed."
    )


if __name__ == "__main__":
    main()