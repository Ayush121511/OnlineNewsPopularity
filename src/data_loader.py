# src/data_loader.py

"""
Data loading and article/feature alignment utilities.

Final feature space
-------------------
Removed completely:
    - six word-based features

Retained:
    - 16 sentiment features
    - 5 LDA features
    - 6 channel features
    - 16 structural/web features

Total:
    43 classifier features

The original UCI values for sentiment and LDA are retained as
ground-truth targets for the reconstruction experiment.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SCRAPED_ARTICLES_PATH,
    RETRIEVAL_METADATA_PATH,
    SENTIMENT_FEATURE_COLUMNS,
    LDA_FEATURE_COLUMNS,
    CHANNEL_FEATURE_COLUMNS,
    STRUCTURAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    SCRAPED_ARTICLE_COLUMNS,
)


# ============================================================
# RAW DATA
# ============================================================

def load_raw_data(
    path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Load the original UCI dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{path}"
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
            "Raw dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    df = df.copy()

    # The original dataframe index is the article ID stored
    # by scrapper.py.
    df["id"] = df.index

    return df


# ============================================================
# PROCESSED DATA
# ============================================================

def load_processed_data(
    path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Load processed_news.csv if it exists."""

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# SCRAPED ARTICLES
# ============================================================

def load_scraped_articles(
    path: Path = SCRAPED_ARTICLES_PATH,
) -> pd.DataFrame:
    """Load successfully scraped article records."""

    if not path.exists():
        raise FileNotFoundError(
            f"Scraped article file not found:\n{path}"
        )

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    missing = [
        column
        for column in SCRAPED_ARTICLE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Scraped article file is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    df = df.copy()

    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce",
    )

    if df["id"].isna().any():
        raise ValueError(
            "Scraped article file contains invalid IDs."
        )

    df["id"] = df["id"].astype(int)

    df["url"] = (
        df["url"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["title"] = (
        df["title"]
        .fillna("")
        .astype(str)
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    return df


# ============================================================
# RETRIEVAL METADATA
# ============================================================

def load_retrieval_metadata(
    path: Path = RETRIEVAL_METADATA_PATH,
) -> pd.DataFrame:
    """Load scraping metadata if available."""

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


# ============================================================
# POPULARITY CLASS
# ============================================================

def assign_popularity_class(
    shares: float,
    class_boundaries: tuple[float, float, float],
) -> int:
    """Assign one of four popularity classes using dataset quartiles."""

    if pd.isna(shares):
        raise ValueError(
            "Cannot classify missing shares."
        )

    if shares <= class_boundaries[0]:
        return 0

    if shares <= class_boundaries[1]:
        return 1

    if shares <= class_boundaries[2]:
        return 2

    return 3


# ============================================================
# MATCHING
# ============================================================

def match_scraped_articles_to_original(
    raw_df: pd.DataFrame,
    scraped_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match scraped articles to original UCI rows using the
    original dataframe index stored as `id`.
    """

    if raw_df.empty:
        raise ValueError(
            "Original dataset is empty."
        )

    if scraped_df.empty:
        raise ValueError(
            "Scraped article dataset is empty."
        )

    if not raw_df["id"].is_unique:
        raise ValueError(
            "Original IDs are not unique."
        )

    if not scraped_df["id"].is_unique:
        raise ValueError(
            "Scraped IDs are not unique."
        )

    merged = scraped_df.merge(
        raw_df,
        on="id",
        how="inner",
        suffixes=("_scraped", "_original"),
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No scraped articles matched the original dataset."
        )

    return merged


# ============================================================
# CANONICAL DATASET
# ============================================================

def build_text_feature_dataset(
    raw_df: pd.DataFrame | None = None,
    scraped_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the canonical dataset used by all reconstruction
    and downstream modeling stages.

    Columns:
        id
        url
        title
        text
        shares
        popularity_class

        16 sentiment features
        5 LDA features
        6 channel features
        16 structural/web features

    The six original word features are absent.
    """

    if raw_df is None:
        raw_df = load_raw_data()

    if scraped_df is None:
        scraped_df = load_scraped_articles()

    merged = match_scraped_articles_to_original(
        raw_df,
        scraped_df,
    )

    # Both datasets contain `url`, therefore after merging
    # pandas normally gives us url_scraped and url_original.
    if "url_scraped" in merged.columns:
        scraped_url_column = "url_scraped"
    elif "url" in merged.columns:
        scraped_url_column = "url"
    else:
        raise ValueError(
            "Scraped URL column not found."
        )

    if "title" not in merged.columns:
        raise ValueError(
            "Scraped title column not found."
        )

    if "text" not in merged.columns:
        raise ValueError(
            "Scraped text column not found."
        )

    # --------------------------------------------------------
    # Validate final modeling columns
    # --------------------------------------------------------

    missing = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column not in merged.columns
    ]

    if missing:
        raise ValueError(
            "Merged dataset is missing modeling features:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    if "shares" not in merged.columns:
        raise ValueError(
            "`shares` column not found."
        )

    # --------------------------------------------------------
    # Construct canonical dataframe
    # --------------------------------------------------------

    dataset = pd.DataFrame()

    dataset["id"] = merged["id"]

    dataset["url"] = merged[
        scraped_url_column
    ]

    dataset["title"] = merged[
        "title"
    ]

    dataset["text"] = merged[
        "text"
    ]

    dataset["shares"] = merged[
        "shares"
    ]

    for column in MODEL_FEATURE_COLUMNS:
        dataset[column] = merged[column]

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    dataset["url"] = (
        dataset["url"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataset["title"] = (
        dataset["title"]
        .fillna("")
        .astype(str)
    )

    dataset["text"] = (
        dataset["text"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Shares
    # --------------------------------------------------------

    dataset["shares"] = pd.to_numeric(
        dataset["shares"],
        errors="coerce",
    )

    # The project report defines four popularity classes using
    # the 25th, 50th, and 75th percentiles of shares.
    # Compute these boundaries from the current full raw dataset
    # rather than hard-coding dataset-specific share values.
    class_boundaries = tuple(
        np.percentile(
            dataset["shares"].dropna().to_numpy(dtype=np.float64),
            [25, 50, 75],
        )
    )

    dataset["popularity_class"] = (
        dataset["shares"].apply(
            lambda shares: assign_popularity_class(
                shares,
                class_boundaries,
            )
        )
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = (
        MODEL_FEATURE_COLUMNS
        + ["shares"]
    )

    for column in numeric_columns:
        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove missing numeric rows
    # --------------------------------------------------------

    invalid_mask = dataset[
        numeric_columns
    ].isna().any(axis=1)

    if invalid_mask.any():

        removed = int(
            invalid_mask.sum()
        )

        print(
            f"Warning: dropping {removed} rows "
            "with missing numeric values."
        )

        dataset = dataset.loc[
            ~invalid_mask
        ].copy()

    # --------------------------------------------------------
    # Remove non-finite rows
    # --------------------------------------------------------

    values = dataset[
        numeric_columns
    ].to_numpy(dtype=np.float64)

    finite_mask = np.isfinite(
        values
    ).all(axis=1)

    if not finite_mask.all():

        removed = int(
            (~finite_mask).sum()
        )

        print(
            f"Warning: dropping {removed} rows "
            "with non-finite numeric values."
        )

        dataset = dataset.loc[
            finite_mask
        ].copy()

    dataset.reset_index(
        drop=True,
        inplace=True,
    )

    return dataset


# ============================================================
# MAIN DATASET LOADER
# ============================================================

def load_feature_prediction_dataset():
    """Load the canonical 996-article reconstruction dataset."""

    dataset = build_text_feature_dataset()

    if dataset.empty:
        raise ValueError(
            "Feature prediction dataset is empty."
        )

    return dataset


# ============================================================
# TARGET HELPERS
# ============================================================

def get_sentiment_targets(
    dataset: pd.DataFrame,
) -> np.ndarray:
    """Return original UCI sentiment targets."""

    return dataset[
        SENTIMENT_FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)


def get_lda_targets(
    dataset: pd.DataFrame,
) -> np.ndarray:
    """Return original UCI LDA targets."""

    return dataset[
        LDA_FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)


def get_reconstruction_targets(
    dataset: pd.DataFrame,
):
    """
    Return the two multi-task target matrices.

    Returns:
        sentiment_targets: (n, 16)
        lda_targets:       (n, 5)
    """

    sentiment_targets = get_sentiment_targets(
        dataset
    )

    lda_targets = get_lda_targets(
        dataset
    )

    return (
        sentiment_targets,
        lda_targets,
    )


# ============================================================
# MATCHING DIAGNOSTICS
# ============================================================

def get_matching_summary(
    raw_df: pd.DataFrame | None = None,
    scraped_df: pd.DataFrame | None = None,
) -> dict:

    if raw_df is None:
        raw_df = load_raw_data()

    if scraped_df is None:
        scraped_df = load_scraped_articles()

    matched = scraped_df[
        scraped_df["id"].isin(
            raw_df["id"]
        )
    ]

    raw_ids = set(
        raw_df["id"]
    )

    scraped_ids = set(
        scraped_df["id"]
    )

    return {
        "original_articles": len(raw_df),
        "scraped_articles": len(scraped_df),
        "matched_articles": len(
            matched["id"].unique()
        ),
        "unmatched_scraped_articles": len(
            scraped_ids - raw_ids
        ),
        "unique_scraped_ids": len(
            scraped_ids
        ),
        "unique_raw_ids": len(
            raw_ids
        ),
    }


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_dataset_diagnostics(
    dataset: pd.DataFrame,
):
    """Print final feature-space statistics."""

    print("=" * 70)
    print("FINAL MODELING DATASET")
    print("=" * 70)

    print(
        f"Articles available : "
        f"{len(dataset):,}"
    )

    print(
        f"Sentiment features : "
        f"{len(SENTIMENT_FEATURE_COLUMNS)}"
    )

    print(
        f"LDA features       : "
        f"{len(LDA_FEATURE_COLUMNS)}"
    )

    print(
        f"Channel features   : "
        f"{len(CHANNEL_FEATURE_COLUMNS)}"
    )

    print(
        f"Structural features: "
        f"{len(STRUCTURAL_FEATURE_COLUMNS)}"
    )

    print(
        f"Total model inputs : "
        f"{len(MODEL_FEATURE_COLUMNS)}"
    )

    print(
        "\nRemoved word features: 6"
    )

    print("\nPopularity classes:")

    counts = (
        dataset["popularity_class"]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():

        percentage = (
            100.0
            * count
            / len(dataset)
        )

        print(
            f"  Class {label}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ONLINE NEWS POPULARITY DATA LOADER")
    print("=" * 70)

    raw_df = load_raw_data()

    scraped_df = load_scraped_articles()

    print(
        f"\nOriginal dataset : "
        f"{len(raw_df):,} rows"
    )

    print(
        f"Scraped articles : "
        f"{len(scraped_df):,} rows"
    )

    summary = get_matching_summary(
        raw_df,
        scraped_df,
    )

    print("\nMatching summary:")

    for key, value in summary.items():
        print(
            f"  {key}: {value:,}"
        )

    dataset = build_text_feature_dataset(
        raw_df,
        scraped_df,
    )

    print()

    print_dataset_diagnostics(
        dataset
    )

    print("\nModeling columns:")

    for index, column in enumerate(
        MODEL_FEATURE_COLUMNS,
        start=1,
    ):
        print(
            f"{index:2d}. {column}"
        )

    sentiment_targets, lda_targets = (
        get_reconstruction_targets(
            dataset
        )
    )

    print(
        "\nSentiment target shape:"
    )
    print(
        f"  {sentiment_targets.shape}"
    )

    print(
        "\nLDA target shape:"
    )
    print(
        f"  {lda_targets.shape}"
    )

    print(
        "\nData loader check: PASSED"
    )


if __name__ == "__main__":
    main()