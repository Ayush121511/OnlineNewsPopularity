"""
config.py

Single source of truth for project configuration.

This file contains:
    - Project/data/model paths
    - Dataset column definitions
    - Text-derived target feature definitions
    - Frozen classifier configuration
    - Scraping configuration
    - Train/validation/test split configuration
    - Transformer configuration
    - LSTM configuration
    - Evaluation configuration
"""

from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================

# src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


# =============================================================================
# DATA FILES
# =============================================================================

RAW_DATA_PATH = DATA_DIR / "OnlineNewsPopularity.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_news.csv"

# New text-retrieval pipeline
SCRAPED_ARTICLES_PATH = DATA_DIR / "scraped_articles.csv"
RETRIEVAL_METADATA_PATH = DATA_DIR / "retrieval_metadata.csv"


# =============================================================================
# MODEL DIRECTORIES
# =============================================================================

CLASSIFIER_MODEL_DIR = MODEL_DIR / "classifiers"
FEATURE_PREDICTOR_MODEL_DIR = MODEL_DIR / "feature_predictors"


# Frozen classifier artifacts
RF_MODEL_PATH = CLASSIFIER_MODEL_DIR / "random_forest.pkl"
SVM_MODEL_PATH = CLASSIFIER_MODEL_DIR / "svm.pkl"
GNB_MODEL_PATH = CLASSIFIER_MODEL_DIR / "gaussian_nb.pkl"
LOGISTIC_MODEL_PATH = CLASSIFIER_MODEL_DIR / "logistic_regression.pkl"


# Text -> feature prediction models
DISTILBERT_MODEL_DIR = FEATURE_PREDICTOR_MODEL_DIR / "distilbert"
LSTM_MODEL_DIR = FEATURE_PREDICTOR_MODEL_DIR / "lstm"

TARGET_STATS_PATH = FEATURE_PREDICTOR_MODEL_DIR / "target_stats.npz"


# =============================================================================
# OUTPUT FILES
# =============================================================================

FEATURE_PREDICTIONS_PATH = OUTPUT_DIR / "feature_predictions.csv"

FEATURE_EVALUATION_PATH = OUTPUT_DIR / "feature_prediction_metrics.csv"

DOWNSTREAM_EVALUATION_PATH = OUTPUT_DIR / "downstream_classifier_metrics.csv"

PREDICTED_CLASSIFIERS_PATH = OUTPUT_DIR / "downstream_predictions.csv"

CONFUSION_MATRIX_DIR = OUTPUT_DIR / "confusion_matrices"


# =============================================================================
# ORIGINAL DATASET COLUMNS
# =============================================================================

URL_COLUMN = "url"
TEXT_COLUMN = "text"
TITLE_COLUMN = "title"

SHARES_COLUMN = "shares"
TARGET_COLUMN = "popularity_class"


# =============================================================================
# TEXT STATISTICS
# =============================================================================
#
# These are properties that can be derived from the article text itself.
#

TEXT_STAT_FEATURE_COLUMNS = (
    "n_tokens_title",
    "n_tokens_content",
    "n_unique_tokens",
    "n_non_stop_words",
    "n_non_stop_unique_tokens",
    "average_token_length",
)


# =============================================================================
# SENTIMENT / SUBJECTIVITY FEATURES
# =============================================================================

SENTIMENT_FEATURE_COLUMNS = (
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
)


# =============================================================================
# LDA TOPIC FEATURES
# =============================================================================

LDA_FEATURE_COLUMNS = (
    "LDA_00",
    "LDA_01",
    "LDA_02",
    "LDA_03",
    "LDA_04",
)


# =============================================================================
# CONTENT CHANNEL FEATURES
# =============================================================================
#
# These are binary indicators representing the article's content channel.
# They are not NLP features in the narrow sense, but they are content/text
# derived and can potentially be inferred from article text.
#

CHANNEL_FEATURE_COLUMNS = (
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
)


# =============================================================================
# ALL TEXT-DERIVED TARGET FEATURES
# =============================================================================
#
# These are the features that the BERT/DistilBERT and LSTM models will
# attempt to predict from raw article text.
#

TEXT_DERIVED_FEATURE_COLUMNS = (
    TEXT_STAT_FEATURE_COLUMNS
    + SENTIMENT_FEATURE_COLUMNS
    + LDA_FEATURE_COLUMNS
    + CHANNEL_FEATURE_COLUMNS
)

TEXT_DERIVED_FEATURE_COUNT = len(TEXT_DERIVED_FEATURE_COLUMNS)


# Backward-compatible alias.
# Prefer TEXT_DERIVED_FEATURE_COLUMNS in all new code.
NLP_FEATURE_COLUMNS = TEXT_DERIVED_FEATURE_COLUMNS


# =============================================================================
# TEXT-DERIVED FEATURE GROUPS
# =============================================================================

TEXT_DERIVED_FEATURE_GROUPS = {
    "text_statistics": TEXT_STAT_FEATURE_COLUMNS,
    "sentiment_subjectivity": SENTIMENT_FEATURE_COLUMNS,
    "lda_topics": LDA_FEATURE_COLUMNS,
    "content_channels": CHANNEL_FEATURE_COLUMNS,
}


# =============================================================================
# FROZEN CLASSIFIER INPUT CONFIGURATION
# =============================================================================
#
# IMPORTANT:
#
# The existing RF/SVM/GNB/Logistic Regression classifiers must remain
# completely unchanged.
#
# We intentionally do NOT hardcode their feature ordering here until the
# exact ordering is recovered from the actual saved classifier artifacts
# / training code.
#
# Once verified, this tuple will contain the exact 58-feature ordering
# expected by the frozen classifiers.
#

DOWNSTREAM_FEATURE_COUNT = 58

FROZEN_CLASSIFIER_FEATURE_COLUMNS = ()

# Features that are allowed to be replaced by predictions from the
# text -> feature models.
#
# This remains empty until the exact frozen classifier schema is verified.
DOWNSTREAM_REPLACEMENT_FEATURES = ()


# =============================================================================
# FEATURES THAT SHOULD REMAIN ORIGINAL
# =============================================================================
#
# These features depend on webpage/article structure rather than plain
# article text and therefore should not be reconstructed by the text model.
#
# Examples from the original dataset include:
#     num_hrefs
#     num_self_hrefs
#     num_imgs
#     num_videos
#     num_keywords
#     keyword statistics
#     self-reference share statistics
#
# This list is informational for now. The exact downstream partition will
# be established after recovering the frozen classifier feature schema.
#

STRUCTURAL_FEATURE_COLUMNS = (
    "num_hrefs",
    "num_self_hrefs",
    "num_imgs",
    "num_videos",
    "num_keywords",
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
)


# =============================================================================
# POPULARITY CLASS CONFIGURATION
# =============================================================================
#
# These values reproduce the class boundaries used in the existing
# preprocessing workflow.
#

NUM_CLASSES = 4

CLASS_BOUNDARIES = (
    916.0,
    1200.0,
    1700.0,
)

CLASS_LABELS = (
    0,
    1,
    2,
    3,
)


# =============================================================================
# SCRAPING CONFIGURATION
# =============================================================================

SCRAPE_SAMPLE_SIZE = 200

SCRAPE_MIN_TEXT_LENGTH = 200

SCRAPE_TIMEOUT_SECONDS = 15

SCRAPE_DELAY_SECONDS = 1.0

SCRAPE_MAX_RETRIES = 2

USE_WAYBACK_FALLBACK = True

WAYBACK_TIMEOUT_SECONDS = 20

WAYBACK_MAX_RETRIES = 2


# =============================================================================
# DATA SPLIT CONFIGURATION
# =============================================================================

RANDOM_SEED = 42

TRAIN_SIZE = 0.80
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10


# =============================================================================
# TRANSFORMER CONFIGURATION
# =============================================================================

TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"

TRANSFORMER_MAX_LENGTH = 512

TRANSFORMER_BATCH_SIZE = 8

TRANSFORMER_LEARNING_RATE = 2e-5

TRANSFORMER_WEIGHT_DECAY = 0.01

TRANSFORMER_EPOCHS = 5

TRANSFORMER_GRADIENT_ACCUMULATION_STEPS = 2

TRANSFORMER_MAX_GRAD_NORM = 1.0

TRANSFORMER_EARLY_STOPPING_PATIENCE = 2


# =============================================================================
# LSTM CONFIGURATION
# =============================================================================

LSTM_EMBEDDING_DIM = 128

LSTM_HIDDEN_DIM = 256

LSTM_NUM_LAYERS = 2

LSTM_DROPOUT = 0.30

LSTM_BIDIRECTIONAL = True

LSTM_BATCH_SIZE = 32

LSTM_LEARNING_RATE = 1e-3

LSTM_WEIGHT_DECAY = 1e-5

LSTM_EPOCHS = 10

LSTM_MAX_SEQUENCE_LENGTH = 512

LSTM_EARLY_STOPPING_PATIENCE = 2


# =============================================================================
# TARGET NORMALIZATION
# =============================================================================
#
# Multi-output regression targets have very different scales.
# Standardizing the targets before MSE training prevents high-magnitude
# targets from dominating the loss.
#

NORMALIZE_TARGETS = True

TARGET_MEAN_FILENAME = "target_mean.npy"

TARGET_STD_FILENAME = "target_std.npy"

TARGET_MEAN_PATH = FEATURE_PREDICTOR_MODEL_DIR / TARGET_MEAN_FILENAME

TARGET_STD_PATH = FEATURE_PREDICTOR_MODEL_DIR / TARGET_STD_FILENAME


# =============================================================================
# LOSS CONFIGURATION
# =============================================================================

LOSS_FUNCTION = "mse"


# =============================================================================
# FEATURE-PREDICTION EVALUATION
# =============================================================================

FEATURE_PREDICTION_METRICS = (
    "r2",
    "mae",
)


# =============================================================================
# DOWNSTREAM CLASSIFIER EVALUATION
# =============================================================================

DOWNSTREAM_METRICS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
)

GENERATE_CONFUSION_MATRICES = True


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

PYTHONHASHSEED = RANDOM_SEED


# =============================================================================
# DIRECTORY CREATION
# =============================================================================

def create_required_directories():
    """
    Create all directories required by the project.
    """

    directories = (
        DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        CLASSIFIER_MODEL_DIR,
        FEATURE_PREDICTOR_MODEL_DIR,
        DISTILBERT_MODEL_DIR,
        LSTM_MODEL_DIR,
        CONFUSION_MATRIX_DIR,
    )

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """
    Validate basic configuration consistency.

    This does not require the datasets or model artifacts to exist.
    """

    if abs(
        TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE - 1.0
    ) > 1e-8:
        raise ValueError(
            "TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE must equal 1.0"
        )

    if NUM_CLASSES != len(CLASS_LABELS):
        raise ValueError(
            "NUM_CLASSES must match the number of CLASS_LABELS."
        )

    if len(CLASS_BOUNDARIES) != NUM_CLASSES - 1:
        raise ValueError(
            "Number of class boundaries must equal NUM_CLASSES - 1."
        )

    if len(TEXT_DERIVED_FEATURE_COLUMNS) != TEXT_DERIVED_FEATURE_COUNT:
        raise ValueError(
            "TEXT_DERIVED_FEATURE_COUNT does not match "
            "TEXT_DERIVED_FEATURE_COLUMNS."
        )

    if DOWNSTREAM_FEATURE_COUNT <= 0:
        raise ValueError(
            "DOWNSTREAM_FEATURE_COUNT must be positive."
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    validate_config()
    create_required_directories()

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data:    {RAW_DATA_PATH}")
    print(f"Processed:   {PROCESSED_DATA_PATH}")

    print()
    print(
        f"Text-derived target features: "
        f"{TEXT_DERIVED_FEATURE_COUNT}"
    )

    for group_name, columns in TEXT_DERIVED_FEATURE_GROUPS.items():
        print(f"  {group_name}: {len(columns)}")

    print()
    print(f"Frozen classifier feature count: {DOWNSTREAM_FEATURE_COUNT}")

    print()
    print("Configuration is valid.")