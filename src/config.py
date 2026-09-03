"""
Central configuration for the Online News Popularity project.
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

RAW_DATA_PATH = DATA_DIR / "OnlineNewsPopularity.csv"

SCRAPED_ARTICLES_PATH = DATA_DIR / "scraped_articles.csv"

RETRIEVAL_METADATA_PATH = DATA_DIR / "retrieval_metadata.csv"

PROCESSED_DATA_PATH = DATA_DIR / "processed_news.csv"


# ============================================================
# SCRAPED ARTICLE COLUMNS
# ============================================================

SCRAPED_ARTICLE_COLUMNS = [
    "id",
    "url",
    "title",
    "text",
]


# ============================================================
# DIRECT TEXT FEATURES
# ============================================================
# The six original word-based features were removed completely
# from the redesigned study.
#
# There are therefore no direct text-derived numeric features
# reconstructed separately from BERT.

DIRECT_TEXT_FEATURE_COLUMNS = []


# ============================================================
# SENTIMENT FEATURES
# ============================================================
# Original UCI sentiment feature values retained as supervised
# reconstruction targets.

SENTIMENT_FEATURE_COLUMNS = [
    "global_subjectivity",
    "global_sentiment_polarity",
    "global_rate_positive_words",
    "global_rate_negative_words",
    "rate_positive_words",
    "rate_negative_words",
    "avg_positive_polarity",
    "min_positive_polarity",
    "max_positive_polarity",
    "avg_negative_polarity",
    "min_negative_polarity",
    "max_negative_polarity",
    "title_subjectivity",
    "title_sentiment_polarity",
    "abs_title_subjectivity",
    "abs_title_sentiment_polarity",
]


# ============================================================
# LDA FEATURES
# ============================================================

LDA_FEATURE_COLUMNS = [
    "LDA_00",
    "LDA_01",
    "LDA_02",
    "LDA_03",
    "LDA_04",
]


# ============================================================
# CHANNEL FEATURES
# ============================================================
# Actual UCI Online News Popularity column names.

CHANNEL_FEATURE_COLUMNS = [
    "data_channel_is_lifestyle",
    "data_channel_is_bus",
    "data_channel_is_entertainment",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
]


# ============================================================
# STRUCTURAL / WEB FEATURES
# ============================================================
# Actual UCI column names.
#
# The six removed word-based features are deliberately excluded.

STRUCTURAL_FEATURE_COLUMNS = [
    "num_hrefs",
    "num_self_hrefs",
    "num_imgs",
    "num_videos",
    "kw_min_min",
    "kw_max_min",
    "kw_avg_min",
    "kw_min_max",
    "kw_max_max",
    "kw_avg_max",
    "kw_min_avg",
    "kw_max_avg",
    "kw_avg_avg",
    "self_reference_min_shares",
    "self_reference_max_shares",
    "self_reference_avg_sharess",
]


# ============================================================
# FINAL MODEL FEATURE SPACE
# ============================================================
#
# 16 sentiment
# + 5 LDA
# + 6 channel
# + 16 structural/web
# = 43 features
#
# The six original word-based features are completely removed.

MODEL_FEATURE_COLUMNS = (
    SENTIMENT_FEATURE_COLUMNS
    + LDA_FEATURE_COLUMNS
    + CHANNEL_FEATURE_COLUMNS
    + STRUCTURAL_FEATURE_COLUMNS
)


# ============================================================
# RECONSTRUCTED TEXT FEATURES
# ============================================================

RECONSTRUCTED_TEXT_FEATURE_COLUMNS = (
    SENTIMENT_FEATURE_COLUMNS
    + LDA_FEATURE_COLUMNS
)


# ============================================================
# FEATURE COUNTS
# ============================================================

DIRECT_TEXT_FEATURE_COUNT = len(
    DIRECT_TEXT_FEATURE_COLUMNS
)

SENTIMENT_FEATURE_COUNT = len(
    SENTIMENT_FEATURE_COLUMNS
)

LDA_FEATURE_COUNT = len(
    LDA_FEATURE_COLUMNS
)

CHANNEL_FEATURE_COUNT = len(
    CHANNEL_FEATURE_COLUMNS
)

STRUCTURAL_FEATURE_COUNT = len(
    STRUCTURAL_FEATURE_COLUMNS
)

MODEL_FEATURE_COUNT = len(
    MODEL_FEATURE_COLUMNS
)


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


# Existing model files use these names.
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
# OUTPUT DIMENSIONS
# ============================================================

SENTIMENT_OUTPUT_DIM = SENTIMENT_FEATURE_COUNT

LDA_OUTPUT_DIM = LDA_FEATURE_COUNT


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
# Four classes using the project percentile boundaries:
#
# <= 916   -> class 0
# <= 1200  -> class 1
# <= 1700  -> class 2
# > 1700   -> class 3

CLASS_BOUNDARIES = (
    916,
    1200,
    1700,
)


# ============================================================
# SCRAPER CONFIGURATION
# ============================================================

SCRAPE_SAMPLE_SIZE = 1000

SCRAPE_TIMEOUT = 15

SCRAPE_SLEEP_SECONDS = 0.5

SCRAPE_USE_WAYBACK = True


# ============================================================
# REQUIRED RAW DATA COLUMNS
# ============================================================
# IMPORTANT:
#
# `title` and `text` come from scraped_articles.csv.
# `popularity_class` is derived from `shares`.
#
# Therefore neither is required here.

REQUIRED_RAW_COLUMNS = [
    "url",
    "shares",

    *SENTIMENT_FEATURE_COLUMNS,

    *LDA_FEATURE_COLUMNS,

    *CHANNEL_FEATURE_COLUMNS,

    *STRUCTURAL_FEATURE_COLUMNS,
]


# ============================================================
# GENERIC FEATURE-PREDICTION ARTIFACTS
# ============================================================

FEATURE_PREDICTOR_MODEL_PATH = (
    MODELS_DIR / "feature_predictor.pt"
)

FEATURE_TARGET_SCALER_PATH = (
    MODELS_DIR / "feature_target_scaler.pkl"
)

FEATURE_SPLIT_INDICES_PATH = (
    OUTPUTS_DIR / "feature_split_indices.npz"
)

FEATURE_PREDICTIONS_PATH = (
    OUTPUTS_DIR / "feature_predictions.csv"
)

FEATURE_PREDICTION_METRICS_PATH = (
    OUTPUTS_DIR / "feature_prediction_metrics.csv"
)


# ============================================================
# SENTIMENT EXPERIMENT ARTIFACTS
# ============================================================

SENTIMENT_MODEL_PATH = (
    MODELS_DIR / "sentiment_predictor.pt"
)

SENTIMENT_TARGET_SCALER_PATH = (
    MODELS_DIR / "sentiment_target_scaler.pkl"
)

SENTIMENT_SPLIT_INDICES_PATH = (
    OUTPUTS_DIR / "sentiment_split_indices.npz"
)

SENTIMENT_PREDICTIONS_PATH = (
    OUTPUTS_DIR / "sentiment_predictions.csv"
)

SENTIMENT_METRICS_PATH = (
    OUTPUTS_DIR / "sentiment_metrics.csv"
)

SENTIMENT_TRAINING_HISTORY_PATH = (
    MODELS_DIR / "sentiment_predictor_training_history.csv"
)


# ============================================================
# LDA EXPERIMENT ARTIFACTS
# ============================================================

LDA_MODEL_PATH = (
    MODELS_DIR / "lda_predictor.pt"
)

LDA_TARGET_SCALER_PATH = (
    MODELS_DIR / "lda_target_scaler.pkl"
)

LDA_SPLIT_INDICES_PATH = (
    OUTPUTS_DIR / "lda_split_indices.npz"
)

LDA_PREDICTIONS_PATH = (
    OUTPUTS_DIR / "lda_predictions.csv"
)

LDA_METRICS_PATH = (
    OUTPUTS_DIR / "lda_metrics.csv"
)

LDA_TRAINING_HISTORY_PATH = (
    MODELS_DIR / "lda_predictor_training_history.csv"
)


# ============================================================
# BERT EMBEDDING ARTIFACTS
# ============================================================

BERT_EMBEDDINGS_PATH = (
    OUTPUTS_DIR / "bert_embeddings.npy"
)

BERT_EMBEDDING_METADATA_PATH = (
    OUTPUTS_DIR / "bert_embedding_metadata.csv"
)

# ============================================================
# SENTIMENT BERT CONFIGURATION
# ============================================================

SENTIMENT_BERT_MODEL_NAME = (
    "cardiffnlp/roberta-base-sentiment"
)

SENTIMENT_BERT_EMBEDDING_DIM = 768

SENTIMENT_BERT_MAX_LENGTH = 512

SENTIMENT_BERT_BATCH_SIZE = 8

SENTIMENT_BERT_EMBEDDINGS_PATH = (
    OUTPUTS_DIR / "sentiment_bert_embeddings.npy"
)

SENTIMENT_BERT_EMBEDDING_METADATA_PATH = (
    OUTPUTS_DIR / "sentiment_bert_embedding_metadata.csv"
)

# ============================================================
# MULTITASK CONFIGURATION
# ============================================================

LDA_LOSS_WEIGHT = 1.0