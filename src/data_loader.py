"""
data_loader.py

Data loading and preparation utilities for the Online News Popularity
text -> feature prediction pipeline.

Responsibilities:
    - Load the original Online News Popularity dataset.
    - Load successfully scraped articles.
    - Load retrieval metadata.
    - Match scraped articles to the original dataset using URL.
    - Construct the supervised text -> text-derived-feature dataset.
    - Validate the resulting data.

This module does NOT:
    - scrape articles
    - train models
    - scale features
    - create popularity classes
    - modify frozen classifier inputs
"""

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SCRAPED_ARTICLES_PATH,
    RETRIEVAL_METADATA_PATH,
    URL_COLUMN,
    TITLE_COLUMN,
    TEXT_COLUMN,
    SHARES_COLUMN,
    TEXT_DERIVED_FEATURE_COLUMNS,
)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def _validate_file_exists(
    path: Path,
    description: str,
) -> None:
    """Raise a clear error if a required file does not exist."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}\n"
            f"Please check the path in config.py."
        )


def _clean_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Strip whitespace from column names.

    Returns a copy and does not modify the original DataFrame.
    """

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def _validate_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Check that all required columns are present."""

    required_columns = list(required_columns)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n"
            f"{missing_columns}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )


# =============================================================================
# ORIGINAL DATASET
# =============================================================================

def load_raw_data(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the original OnlineNewsPopularity.csv dataset.

    No preprocessing is performed here.

    Parameters
    ----------
    path:
        Optional path overriding RAW_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Original dataset.
    """

    path = Path(path) if path is not None else RAW_DATA_PATH

    _validate_file_exists(
        path,
        "Raw dataset",
    )

    df = pd.read_csv(path)

    df = _clean_column_names(df)

    _validate_columns(
        df,
        [
            URL_COLUMN,
            SHARES_COLUMN,
            *TEXT_DERIVED_FEATURE_COLUMNS,
        ],
        "Raw dataset",
    )

    return df


# =============================================================================
# PROCESSED DATASET
# =============================================================================

def load_processed_data(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the existing processed dataset.

    This function only loads the file. It does not reproduce or modify
    the old preprocessing pipeline.
    """

    path = (
        Path(path)
        if path is not None
        else PROCESSED_DATA_PATH
    )

    _validate_file_exists(
        path,
        "Processed dataset",
    )

    df = pd.read_csv(path)

    df = _clean_column_names(df)

    return df


# =============================================================================
# SCRAPED ARTICLES
# =============================================================================

def load_scraped_articles(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load successfully scraped articles.

    Expected columns:
        id
        url
        title
        text

    Additional columns are allowed.
    """

    path = (
        Path(path)
        if path is not None
        else SCRAPED_ARTICLES_PATH
    )

    _validate_file_exists(
        path,
        "Scraped article dataset",
    )

    df = pd.read_csv(path)

    df = _clean_column_names(df)

    _validate_columns(
        df,
        [
            "id",
            URL_COLUMN,
            TITLE_COLUMN,
            TEXT_COLUMN,
        ],
        "Scraped article dataset",
    )

    # Normalize string fields.
    df[URL_COLUMN] = (
        df[URL_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df[TITLE_COLUMN] = (
        df[TITLE_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove rows without usable URLs or text.
    df = df[
        (df[URL_COLUMN] != "")
        & (df[TEXT_COLUMN] != "")
    ].copy()

    # Keep only one scraped record per URL.
    df = df.drop_duplicates(
        subset=[URL_COLUMN],
        keep="first",
    )

    return df.reset_index(drop=True)


# =============================================================================
# RETRIEVAL METADATA
# =============================================================================

def load_retrieval_metadata(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load metadata describing article retrieval attempts.

    Failed retrievals remain represented here even though they do not
    appear in scraped_articles.csv.
    """

    path = (
        Path(path)
        if path is not None
        else RETRIEVAL_METADATA_PATH
    )

    _validate_file_exists(
        path,
        "Retrieval metadata",
    )

    df = pd.read_csv(path)

    df = _clean_column_names(df)

    return df


# =============================================================================
# TARGET FEATURES
# =============================================================================

def get_text_derived_target_columns() -> Tuple[str, ...]:
    """Return the complete set of text-derived prediction targets."""

    return tuple(TEXT_DERIVED_FEATURE_COLUMNS)


def extract_text_derived_targets(
    df: pd.DataFrame,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """
    Extract the text-derived target features from the original dataset.

    These are the ground-truth targets for the DistilBERT/BERT and
    LSTM models.
    """

    target_columns = (
        get_text_derived_target_columns()
    )

    _validate_columns(
        df,
        target_columns,
        "Text-derived target dataset",
    )

    targets = df[
        list(target_columns)
    ].copy()

    if drop_missing:
        targets = targets.dropna(
            subset=list(target_columns)
        ).reset_index(drop=True)

    return targets


# =============================================================================
# URL NORMALIZATION
# =============================================================================

def normalize_url(url: str) -> str:
    """
    Normalize a URL for matching.

    This is intentionally conservative. We do not alter the URL's
    semantic path or query parameters.
    """

    if pd.isna(url):
        return ""

    url = str(url).strip()

    if not url:
        return ""

    # Remove a trailing slash only.
    if url.endswith("/"):
        url = url[:-1]

    return url


# =============================================================================
# BUILD SUPERVISED DATASET
# =============================================================================

def build_text_target_dataset(
    articles: pd.DataFrame,
    original_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match scraped article text with ground-truth text-derived features.

    Matching is performed using URL.

    The original feature values are NOT recomputed from the scraped text.
    They are treated as ground-truth supervised targets.

    Output contains:

        id
        url
        title
        text
        shares
        <32 text-derived features>

    Parameters
    ----------
    articles:
        Successfully scraped articles.

    original_data:
        Original OnlineNewsPopularity dataset.

    Returns
    -------
    pd.DataFrame
        Supervised text -> feature dataset.
    """

    # -------------------------------------------------------------------------
    # Validate input schemas
    # -------------------------------------------------------------------------

    _validate_columns(
        articles,
        [
            "id",
            URL_COLUMN,
            TITLE_COLUMN,
            TEXT_COLUMN,
        ],
        "Scraped articles",
    )

    _validate_columns(
        original_data,
        [
            URL_COLUMN,
            SHARES_COLUMN,
            *TEXT_DERIVED_FEATURE_COLUMNS,
        ],
        "Original dataset",
    )

    # -------------------------------------------------------------------------
    # Work on copies
    # -------------------------------------------------------------------------

    articles = articles.copy()
    original_data = original_data.copy()

    # -------------------------------------------------------------------------
    # Normalize URLs
    # -------------------------------------------------------------------------

    articles["_match_url"] = (
        articles[URL_COLUMN]
        .map(normalize_url)
    )

    original_data["_match_url"] = (
        original_data[URL_COLUMN]
        .map(normalize_url)
    )

    # Remove unusable URLs.
    articles = articles[
        articles["_match_url"] != ""
    ].copy()

    original_data = original_data[
        original_data["_match_url"] != ""
    ].copy()

    # -------------------------------------------------------------------------
    # Remove duplicate URLs
    # -------------------------------------------------------------------------

    articles = articles.drop_duplicates(
        subset=["_match_url"],
        keep="first",
    )

    original_data = original_data.drop_duplicates(
        subset=["_match_url"],
        keep="first",
    )

    # -------------------------------------------------------------------------
    # Select only required ground-truth columns
    # -------------------------------------------------------------------------

    target_columns = [
        URL_COLUMN,
        SHARES_COLUMN,
        *TEXT_DERIVED_FEATURE_COLUMNS,
        "_match_url",
    ]

    original_targets = original_data[
        target_columns
    ].copy()

    # -------------------------------------------------------------------------
    # URL join
    # -------------------------------------------------------------------------

    merged = articles.merge(
        original_targets,
        on="_match_url",
        how="inner",
        suffixes=("_scraped", "_original"),
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No scraped articles could be matched with the original "
            "dataset using URL."
        )

    # -------------------------------------------------------------------------
    # Resolve URL column
    # -------------------------------------------------------------------------

    # Keep the original URL from the scraped dataset because this is
    # the URL that was actually used for retrieval.
    if f"{URL_COLUMN}_scraped" in merged.columns:
        merged[URL_COLUMN] = merged[
            f"{URL_COLUMN}_scraped"
        ]

    # -------------------------------------------------------------------------
    # Resolve title
    # -------------------------------------------------------------------------

    if f"{TITLE_COLUMN}_scraped" in merged.columns:
        merged[TITLE_COLUMN] = merged[
            f"{TITLE_COLUMN}_scraped"
        ]

    # -------------------------------------------------------------------------
    # Clean text
    # -------------------------------------------------------------------------

    merged[TEXT_COLUMN] = (
        merged[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    merged[TITLE_COLUMN] = (
        merged[TITLE_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove articles without usable text.
    merged = merged[
        merged[TEXT_COLUMN].str.len() > 0
    ].copy()

    # -------------------------------------------------------------------------
    # Remove missing target values
    # -------------------------------------------------------------------------

    merged = merged.dropna(
        subset=list(TEXT_DERIVED_FEATURE_COLUMNS)
    ).copy()

    # -------------------------------------------------------------------------
    # Keep only columns needed by the feature prediction pipeline
    # -------------------------------------------------------------------------

    output_columns = [
        "id",
        URL_COLUMN,
        TITLE_COLUMN,
        TEXT_COLUMN,
        SHARES_COLUMN,
        *TEXT_DERIVED_FEATURE_COLUMNS,
    ]

    # Only retain columns that exist.
    output_columns = [
        column
        for column in output_columns
        if column in merged.columns
    ]

    merged = merged[
        output_columns
    ].copy()

    # -------------------------------------------------------------------------
    # Final cleanup
    # -------------------------------------------------------------------------

    merged = merged.drop_duplicates(
        subset=[URL_COLUMN],
        keep="first",
    )

    merged = merged.reset_index(drop=True)

    return merged


# =============================================================================
# FINAL PIPELINE DATASET
# =============================================================================

def load_feature_prediction_dataset() -> pd.DataFrame:
    """
    Load and construct the final supervised dataset for the
    text -> feature prediction experiment.

    Returns
    -------
    pd.DataFrame
        Scraped article text paired with ground-truth text-derived features.
    """

    original_data = load_raw_data()

    scraped_articles = load_scraped_articles()

    dataset = build_text_target_dataset(
        articles=scraped_articles,
        original_data=original_data,
    )

    validate_text_derived_targets(dataset)

    return dataset


# =============================================================================
# VALIDATION
# =============================================================================

def validate_text_derived_targets(
    df: pd.DataFrame,
) -> None:
    """
    Validate that all text-derived targets are numeric and finite.
    """

    target_columns = (
        get_text_derived_target_columns()
    )

    _validate_columns(
        df,
        target_columns,
        "Text-derived target dataset",
    )

    for column in target_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            raise TypeError(
                f"Target '{column}' must be numeric, "
                f"but has dtype {df[column].dtype}."
            )

    values = df[
        list(target_columns)
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Text-derived targets contain NaN or infinite values."
        )


# =============================================================================
# DATASET DESCRIPTION
# =============================================================================

def describe_dataset(
    df: pd.DataFrame,
    name: str = "Dataset",
) -> None:
    """
    Print basic diagnostic information about a dataset.
    """

    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    print()

    print("Missing values:")

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ]

    if missing.empty:
        print("  None")
    else:
        print(
            missing
            .sort_values(ascending=False)
            .to_string()
        )

    print()


# =============================================================================
# MATCHING DIAGNOSTICS
# =============================================================================

def print_matching_summary() -> None:
    """
    Print how many scraped articles successfully match the original
    dataset and how many retrievals succeeded/failed.
    """

    original_data = load_raw_data()

    scraped_articles = load_scraped_articles()

    metadata = load_retrieval_metadata()

    original_urls = set(
        original_data[URL_COLUMN]
        .map(normalize_url)
    )

    scraped_urls = set(
        scraped_articles[URL_COLUMN]
        .map(normalize_url)
    )

    matched_urls = (
        scraped_urls
        & original_urls
    )

    print("=" * 70)
    print("SCRAPING / DATA MATCHING SUMMARY")
    print("=" * 70)

    print(
        f"Original dataset articles : "
        f"{len(original_data):,}"
    )

    print(
        f"Scraped articles          : "
        f"{len(scraped_articles):,}"
    )

    print(
        f"URLs matched              : "
        f"{len(matched_urls):,}"
    )

    if not metadata.empty:

        if "status" in metadata.columns:

            successful = (
                metadata["status"] == "success"
            ).sum()

            failed = (
                metadata["status"] == "failed"
            ).sum()

            print(
                f"Retrieval successes      : "
                f"{successful:,}"
            )

            print(
                f"Retrieval failures       : "
                f"{failed:,}"
            )

        if "source" in metadata.columns:

            direct = (
                metadata["source"] == "direct"
            ).sum()

            wayback = (
                metadata["source"] == "wayback"
            ).sum()

            print(
                f"Direct retrievals        : "
                f"{direct:,}"
            )

            print(
                f"Wayback retrievals       : "
                f"{wayback:,}"
            )

    print()


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DATA LOADER TEST")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load original data
    # -------------------------------------------------------------------------

    original = load_raw_data()

    print(
        f"Original dataset loaded: "
        f"{len(original):,} rows"
    )

    # -------------------------------------------------------------------------
    # Load scraped data
    # -------------------------------------------------------------------------

    scraped = load_scraped_articles()

    print(
        f"Scraped articles loaded: "
        f"{len(scraped):,} rows"
    )

    # -------------------------------------------------------------------------
    # Build supervised dataset
    # -------------------------------------------------------------------------

    dataset = build_text_target_dataset(
        articles=scraped,
        original_data=original,
    )

    print()

    describe_dataset(
        dataset,
        "TEXT -> FEATURE DATASET",
    )

    # -------------------------------------------------------------------------
    # Validate targets
    # -------------------------------------------------------------------------

    validate_text_derived_targets(
        dataset
    )

    # -------------------------------------------------------------------------
    # Print target information
    # -------------------------------------------------------------------------

    print(
        f"Text-derived target count: "
        f"{len(TEXT_DERIVED_FEATURE_COLUMNS)}"
    )

    print()

    print("Target features:")

    for i, column in enumerate(
        TEXT_DERIVED_FEATURE_COLUMNS,
        start=1,
    ):
        print(
            f"  {i:2d}. {column}"
        )

    print()

    print(
        "data_loader.py test completed successfully."
    )