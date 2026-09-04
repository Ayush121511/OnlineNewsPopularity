"""
Central configuration for the Reddit Popularity project.

(Adapted from the Online News Popularity / Mashable project. This
version points at the Reddit comparison dataset, which already
contains text + meta features + label in a single CSV — no separate
scrape/merge step is needed.)
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA FILES
# ============================================================
# Reddit dataset is already fully built (id, title, selftext, text,
# score, popularity_class, meta features) — no scraping/merge step.

RAW_DATA_PATH = DATA_DIR / "reddit_popularity_dataset.csv"


# ============================================================
# META / STRUCTURAL FEATURES
# ============================================================
# Mirrors the role of the Mashable 43-feature tabular set: everything
# that is NOT the free-text title/selftext.

META_FEATURE_COLUMNS = [
    "n_tokens_title",
    "n_tokens_selftext",
    "has_selftext",
    "is_self",
    "over_18",
    "has_thumbnail",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "domain_is_self",
    "domain_is_imgur",
    "domain_is_youtube",
    "title_has_question",
    "title_has_exclaim",
    "title_num_caps_words",
    "num_comments",
    "subreddit_code",
]

# Kept as an alias so any old code importing MODEL_FEATURE_COLUMNS
# still works without edits.
MODEL_FEATURE_COLUMNS = META_FEATURE_COLUMNS

MODEL_FEATURE_COUNT = len(MODEL_FEATURE_COLUMNS)


# ============================================================
# REQUIRED RAW DATA COLUMNS
# ============================================================

REQUIRED_RAW_COLUMNS = [
    "id",
    "subreddit",
    "title",
    "selftext",
    "text",
    "score",
    "score_pct_within_sub",
    "popularity_class",
    *META_FEATURE_COLUMNS,
]


# ============================================================
# BERT CONFIGURATION
# ============================================================

BERT_MODEL_NAME = "distilbert-base-uncased"

BERT_EMBEDDING_DIM = 768

BERT_MAX_LENGTH = 512

BERT_BATCH_SIZE = 8


# ============================================================
# FNN ARCHITECTURE
# ============================================================

FNN_HIDDEN_DIM_1 = 256

FNN_HIDDEN_DIM_2 = 128

FNN_DROPOUT = 0.30

FNN_SHARED_DIM_1 = FNN_HIDDEN_DIM_1

FNN_SHARED_DIM_2 = FNN_HIDDEN_DIM_2


# ============================================================
# FNN TRAINING
# ============================================================

FNN_BATCH_SIZE = 32

FNN_LEARNING_RATE = 1e-3

FNN_WEIGHT_DECAY = 1e-5

FNN_MAX_EPOCHS = 200

FNN_PATIENCE = 20

FNN_MIN_DELTA = 1e-5


# ============================================================
# TARGET PREPROCESSING
# ============================================================

TARGET_STANDARDIZATION = True


# ============================================================
# DATA SPLIT
# ============================================================

RANDOM_SEED = 42

TRAIN_SIZE = 0.80

VALIDATION_SIZE = 0.10

TEST_SIZE = 0.10


# ============================================================
# POPULARITY CLASSES
# ============================================================
# 4 classes, computed at dataset-build time as within-subreddit
# score percentile -> global quartile (see build_reddit_dataset.py).
# `popularity_class` in the CSV is already final; nothing to
# recompute here.

NUM_CLASSES = 4


# ============================================================
# BERT EMBEDDING ARTIFACTS
# ============================================================

BERT_EMBEDDINGS_PATH = (
    OUTPUTS_DIR / "bert_embeddings.npy"
)

BERT_EMBEDDING_METADATA_PATH = (
    OUTPUTS_DIR / "bert_embedding_metadata.csv"
)