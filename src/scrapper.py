"""
scraper.py

Article retrieval pipeline for the Online News Popularity project.

Workflow:
    1. Load the original OnlineNewsPopularity dataset.
    2. Randomly sample a small number of articles.
    3. Try to retrieve each article using newspaper3k.
    4. If direct retrieval fails, optionally try the Wayback Machine.
    5. Save successfully retrieved title + article text.
    6. Save retrieval metadata for every attempted article.

The scraper is intentionally independent of model training/evaluation.
"""

import random
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from newspaper import Article
from newspaper.article import ArticleException

from config import (
    RAW_DATA_PATH,
    SCRAPED_ARTICLES_PATH,
    RETRIEVAL_METADATA_PATH,
    URL_COLUMN,
    TITLE_COLUMN,
    TEXT_COLUMN,
    SCRAPE_SAMPLE_SIZE,
    SCRAPE_MIN_TEXT_LENGTH,
    SCRAPE_TIMEOUT_SECONDS,
    SCRAPE_DELAY_SECONDS,
    SCRAPE_MAX_RETRIES,
    USE_WAYBACK_FALLBACK,
    WAYBACK_TIMEOUT_SECONDS,
    WAYBACK_MAX_RETRIES,
    RANDOM_SEED,
)


# =============================================================================
# USER AGENT
# =============================================================================

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_source_data() -> pd.DataFrame:
    """
    Load the original OnlineNewsPopularity dataset.

    Returns
    -------
    pd.DataFrame
        Original dataset containing article URLs.
    """

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    df.columns = [str(column).strip() for column in df.columns]

    if URL_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{URL_COLUMN}' was not found in the raw dataset."
        )

    return df


# =============================================================================
# SAMPLING
# =============================================================================

def select_articles(
    df: pd.DataFrame,
    sample_size: int = SCRAPE_SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Randomly select articles for scraping.

    The random seed makes the sample reproducible.

    Parameters
    ----------
    df:
        Original dataset.

    sample_size:
        Number of articles to select.

    Returns
    -------
    pd.DataFrame
        Sampled articles with an internal article ID.
    """

    data = df.copy()

    # Remove rows without URLs.
    data = data.dropna(subset=[URL_COLUMN]).copy()

    # Normalize URL values.
    data[URL_COLUMN] = data[URL_COLUMN].astype(str).str.strip()

    data = data[data[URL_COLUMN] != ""]

    # Remove duplicate URLs.
    data = data.drop_duplicates(
        subset=[URL_COLUMN],
        keep="first",
    )

    sample_size = min(sample_size, len(data))

    sampled = data.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    ).copy()

    # Preserve the original dataframe index as the article identifier.
    sampled.insert(
        0,
        "id",
        sampled.index.astype(int),
    )

    sampled = sampled.reset_index(drop=True)

    return sampled


# =============================================================================
# NEWSPAPER3K DIRECT RETRIEVAL
# =============================================================================

def scrape_with_newspaper(
    url: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Attempt to retrieve an article using newspaper3k.

    Returns
    -------
    title:
        Article title if extraction succeeds.

    text:
        Article body if extraction succeeds.

    error:
        Error description if retrieval fails.
    """

    last_error = None

    for attempt in range(SCRAPE_MAX_RETRIES + 1):

        try:
            article = Article(
                url,
                request_timeout=SCRAPE_TIMEOUT_SECONDS,
                browser_user_agent=USER_AGENT,
            )

            article.download()
            article.parse()

            title = (article.title or "").strip()
            text = (article.text or "").strip()

            if len(text) < SCRAPE_MIN_TEXT_LENGTH:
                return (
                    None,
                    None,
                    f"text_too_short:{len(text)}",
                )

            return title, text, None

        except Exception as exc:
            last_error = str(exc)

            if attempt < SCRAPE_MAX_RETRIES:
                time.sleep(SCRAPE_DELAY_SECONDS)

    return None, None, last_error


# =============================================================================
# WAYBACK MACHINE
# =============================================================================

def get_wayback_url(
    url: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Ask the Wayback Machine CDX API for the closest available
    archived version of a URL.

    Returns
    -------
    archived_url:
        URL of the archived page.

    error:
        Error description if no usable snapshot is found.
    """

    cdx_url = "https://web.archive.org/cdx/search/cdx"

    params = {
        "url": url,
        "output": "json",
        "filter": "statuscode:200",
        "filter": "mimetype:text/html",
        "collapse": "digest",
        "limit": 1,
        "fl": "timestamp,original,statuscode",
    }

    headers = {
        "User-Agent": USER_AGENT,
    }

    last_error = None

    for attempt in range(WAYBACK_MAX_RETRIES + 1):

        try:
            response = requests.get(
                cdx_url,
                params=params,
                headers=headers,
                timeout=WAYBACK_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            data = response.json()

            if len(data) < 2:
                return None, "no_wayback_snapshot"

            # First row contains column names.
            timestamp, original_url, status_code = data[1]

            if status_code != "200":
                return None, f"wayback_status:{status_code}"

            archived_url = (
                f"https://web.archive.org/web/"
                f"{timestamp}id_/"
                f"{original_url}"
            )

            return archived_url, None

        except Exception as exc:
            last_error = str(exc)

            if attempt < WAYBACK_MAX_RETRIES:
                time.sleep(SCRAPE_DELAY_SECONDS)

    return None, last_error


def scrape_wayback(
    url: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Retrieve an article from a Wayback Machine snapshot using newspaper3k.

    Returns
    -------
    title, text, error
    """

    archived_url, error = get_wayback_url(url)

    if archived_url is None:
        return None, None, error

    title, text, newspaper_error = scrape_with_newspaper(
        archived_url
    )

    if newspaper_error is not None:
        return None, None, newspaper_error

    return title, text, None


# =============================================================================
# EXISTING RESULTS
# =============================================================================

def load_existing_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved scraping results.

    This makes the scraper resumable.

    Returns
    -------
    articles:
        Previously successful articles.

    metadata:
        Previously attempted articles.
    """

    if SCRAPED_ARTICLES_PATH.exists():
        articles = pd.read_csv(SCRAPED_ARTICLES_PATH)

        if "id" not in articles.columns:
            articles = pd.DataFrame()

    else:
        articles = pd.DataFrame()

    if RETRIEVAL_METADATA_PATH.exists():
        metadata = pd.read_csv(RETRIEVAL_METADATA_PATH)

    else:
        metadata = pd.DataFrame()

    return articles, metadata


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    articles: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    """
    Save successful articles and retrieval metadata.
    """

    SCRAPED_ARTICLES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RETRIEVAL_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    articles.to_csv(
        SCRAPED_ARTICLES_PATH,
        index=False,
    )

    metadata.to_csv(
        RETRIEVAL_METADATA_PATH,
        index=False,
    )


# =============================================================================
# SINGLE ARTICLE
# =============================================================================

def retrieve_article(
    article_id: int,
    url: str,
) -> Tuple[
    Optional[str],
    Optional[str],
    str,
    Optional[str],
]:
    """
    Retrieve one article.

    Returns
    -------
    title
    text
    source
    error
    """

    # -------------------------------------------------------------------------
    # Direct newspaper3k retrieval
    # -------------------------------------------------------------------------

    title, text, error = scrape_with_newspaper(url)

    if text is not None:
        return title, text, "direct", None

    direct_error = error

    # -------------------------------------------------------------------------
    # Wayback fallback
    # -------------------------------------------------------------------------

    if USE_WAYBACK_FALLBACK:

        title, text, error = scrape_wayback(url)

        if text is not None:
            return title, text, "wayback", None

        wayback_error = error

        combined_error = (
            f"direct={direct_error}; "
            f"wayback={wayback_error}"
        )

        return None, None, "failed", combined_error

    return None, None, "failed", direct_error


# =============================================================================
# MAIN SCRAPING PIPELINE
# =============================================================================

def run_scraper() -> None:
    """
    Execute the complete article retrieval pipeline.
    """

    print("=" * 70)
    print("ARTICLE SCRAPING")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load source dataset
    # -------------------------------------------------------------------------

    df = load_source_data()

    print(f"Total articles in dataset : {len(df):,}")

    # -------------------------------------------------------------------------
    # Select reproducible sample
    # -------------------------------------------------------------------------

    sampled = select_articles(
        df,
        sample_size=SCRAPE_SAMPLE_SIZE,
    )

    print(f"Articles selected          : {len(sampled):,}")
    print()

    # -------------------------------------------------------------------------
    # Load previous results
    # -------------------------------------------------------------------------

    existing_articles, existing_metadata = load_existing_results()

    successful_ids = set()

    if not existing_articles.empty and "id" in existing_articles.columns:
        successful_ids = set(
            existing_articles["id"].astype(int)
        )

    attempted_ids = set()

    if not existing_metadata.empty and "id" in existing_metadata.columns:
        attempted_ids = set(
            existing_metadata["id"].astype(int)
        )

    # -------------------------------------------------------------------------
    # Initialize result lists
    # -------------------------------------------------------------------------

    article_records = []

    if not existing_articles.empty:
        article_records = existing_articles.to_dict(
            orient="records"
        )

    metadata_records = []

    if not existing_metadata.empty:
        metadata_records = existing_metadata.to_dict(
            orient="records"
        )

    # -------------------------------------------------------------------------
    # Process each article
    # -------------------------------------------------------------------------

    for position, row in enumerate(
        sampled.itertuples(index=False),
        start=1,
    ):

        article_id = int(row.id)
        url = str(getattr(row, URL_COLUMN)).strip()

        print(
            f"[{position}/{len(sampled)}] "
            f"Article ID: {article_id}"
        )

        # -------------------------------------------------------------
        # Skip already successful article
        # -------------------------------------------------------------

        if article_id in successful_ids:
            print("  Status: already scraped")
            print()
            continue

        # -------------------------------------------------------------
        # Skip previously attempted failures
        # -------------------------------------------------------------

        if article_id in attempted_ids:
            print("  Status: already attempted")
            print()
            continue

        # -------------------------------------------------------------
        # Retrieve
        # -------------------------------------------------------------

        start_time = time.time()

        title, text, source, error = retrieve_article(
            article_id,
            url,
        )

        elapsed = time.time() - start_time

        # -------------------------------------------------------------
        # Success
        # -------------------------------------------------------------

        if text is not None:

            article_records.append(
                {
                    "id": article_id,
                    URL_COLUMN: url,
                    TITLE_COLUMN: title or "",
                    TEXT_COLUMN: text,
                }
            )

            metadata_records.append(
                {
                    "id": article_id,
                    URL_COLUMN: url,
                    "status": "success",
                    "source": source,
                    "text_length": len(text),
                    "elapsed_seconds": round(elapsed, 3),
                    "error": "",
                }
            )

            successful_ids.add(article_id)

            print(
                f"  Status: SUCCESS ({source})"
            )

            print(
                f"  Text length: {len(text):,}"
            )

        # -------------------------------------------------------------
        # Failure
        # -------------------------------------------------------------

        else:

            metadata_records.append(
                {
                    "id": article_id,
                    URL_COLUMN: url,
                    "status": "failed",
                    "source": source,
                    "text_length": 0,
                    "elapsed_seconds": round(elapsed, 3),
                    "error": error or "unknown_error",
                }
            )

            print("  Status: FAILED")
            print(f"  Error: {error}")

        # -------------------------------------------------------------
        # Delay between articles
        # -------------------------------------------------------------

        if position < len(sampled):
            time.sleep(SCRAPE_DELAY_SECONDS)

        # -------------------------------------------------------------
        # Save after every article
        # -------------------------------------------------------------

        articles_df = pd.DataFrame(article_records)
        metadata_df = pd.DataFrame(metadata_records)

        save_results(
            articles_df,
            metadata_df,
        )

        print()

    # -------------------------------------------------------------------------
    # Final results
    # -------------------------------------------------------------------------

    articles_df = pd.DataFrame(article_records)
    metadata_df = pd.DataFrame(metadata_records)

    save_results(
        articles_df,
        metadata_df,
    )

    print("=" * 70)
    print("SCRAPING COMPLETE")
    print("=" * 70)

    print(
        f"Selected articles : {len(sampled):,}"
    )

    print(
        f"Successful articles: {len(articles_df):,}"
    )

    if not metadata_df.empty:

        successful = (
            metadata_df["status"] == "success"
        ).sum()

        failed = (
            metadata_df["status"] == "failed"
        ).sum()

        direct = (
            metadata_df["source"] == "direct"
        ).sum()

        wayback = (
            metadata_df["source"] == "wayback"
        ).sum()

        print(f"Direct successes  : {direct:,}")
        print(f"Wayback successes : {wayback:,}")
        print(f"Failures          : {failed:,}")

    print()
    print(
        f"Articles saved to:\n"
        f"{SCRAPED_ARTICLES_PATH}"
    )

    print()
    print(
        f"Metadata saved to:\n"
        f"{RETRIEVAL_METADATA_PATH}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_scraper()