# src/data_loader.py

"""
Data loading for the Reddit popularity comparison dataset.

Unlike the Mashable pipeline, this dataset already contains text,
meta/structural features, and the popularity label in a single CSV
(built by build_reddit_dataset.py) — no scrape/merge step needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RAW_DATA_PATH,
    REQUIRED_RAW_COLUMNS,
    META_FEATURE_COLUMNS,
    NUM_CLASSES,
)


# ============================================================
# RAW DATA
# ============================================================

def load_raw_data(
    path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Load the Reddit popularity dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}\n"
            "Run build_reddit_dataset.py first."
        )

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    missing = [
        column
        for column in REQUIRED_RAW_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    return df.copy()


# ============================================================
# CANONICAL DATASET
# ============================================================

def build_feature_dataset(
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the canonical dataset used by downstream modeling stages.

    Columns:
        id, subreddit, title, selftext, text,
        score, popularity_class,
        + META_FEATURE_COLUMNS
    """

    if raw_df is None:
        raw_df = load_raw_data()

    dataset = raw_df.copy()

    dataset["title"] = dataset["title"].fillna("").astype(str)
    dataset["selftext"] = dataset["selftext"].fillna("").astype(str)
    dataset["text"] = dataset["text"].fillna("").astype(str)

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = META_FEATURE_COLUMNS + [
        "score",
        "score_pct_within_sub",
        "popularity_class",
    ]

    for column in numeric_columns:
        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove missing / non-finite numeric rows
    # --------------------------------------------------------

    invalid_mask = dataset[numeric_columns].isna().any(axis=1)

    if invalid_mask.any():
        removed = int(invalid_mask.sum())
        print(
            f"Warning: dropping {removed} rows "
            "with missing numeric values."
        )
        dataset = dataset.loc[~invalid_mask].copy()

    values = dataset[numeric_columns].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(values).all(axis=1)

    if not finite_mask.all():
        removed = int((~finite_mask).sum())
        print(
            f"Warning: dropping {removed} rows "
            "with non-finite numeric values."
        )
        dataset = dataset.loc[finite_mask].copy()

    dataset["popularity_class"] = dataset["popularity_class"].astype(int)

    dataset.reset_index(drop=True, inplace=True)

    return dataset


# ============================================================
# MAIN DATASET LOADER
# ============================================================

def load_feature_prediction_dataset() -> pd.DataFrame:
    """Load the canonical Reddit popularity dataset."""

    dataset = build_feature_dataset()

    if dataset.empty:
        raise ValueError(
            "Feature prediction dataset is empty."
        )

    return dataset


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_dataset_diagnostics(dataset: pd.DataFrame):
    """Print final feature-space statistics."""

    print("=" * 70)
    print("FINAL MODELING DATASET (Reddit)")
    print("=" * 70)

    print(f"Rows available     : {len(dataset):,}")
    print(f"Meta features       : {len(META_FEATURE_COLUMNS)}")
    print(f"Subreddits          : {dataset['subreddit'].nunique()}")

    print("\nPopularity classes:")
    counts = (
        dataset["popularity_class"]
        .value_counts()
        .sort_index()
    )
    for label, count in counts.items():
        percentage = 100.0 * count / len(dataset)
        print(f"  Class {label}: {count:,} ({percentage:.2f}%)")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("REDDIT POPULARITY DATA LOADER")
    print("=" * 70)

    dataset = load_feature_prediction_dataset()

    print(f"\nRows loaded: {len(dataset):,}")

    print()
    print_dataset_diagnostics(dataset)

    print("\nModeling columns:")
    for index, column in enumerate(META_FEATURE_COLUMNS, start=1):
        print(f"{index:2d}. {column}")

    print("\nData loader check: PASSED")


if __name__ == "__main__":
    main()