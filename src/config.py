# src/config.py

"""
Central configuration for the Online News Popularity project.

Final methodology
------------------
1. The six original word-based features are completely removed.

2. Sixteen original sentiment features are reconstructed using a
   frozen DistilBERT representation and a supervised neural-network
   head.

3. Five original UCI LDA features are reconstructed using the same
   frozen DistilBERT representation and a second neural-network head.

4. The sentiment and LDA tasks share the same learned FNN backbone.

5. Channel and structural/web features are retained from the original
   dataset.

6. Popularity classifiers are retrained on the resulting feature space.

BERT is NEVER fine-tuned.
"""

from pathlib import Path


# ============================================================
# PROJECT DIRECTORIES
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

PROCESSED_DATA_PATH = DATA_DIR / "processed_news.csv"

SCRAPED_ARTICLES_PATH = DATA_DIR / "scraped_articles.csv"

RETRIEVAL_METADATA_PATH = DATA_DIR / "retrieval_metadata.csv"


# ============================================================
# BERT OUTPUTS
# ============================================================

BERT_EMBEDDINGS_PATH = (
    OUTPUTS_DIR / "bert_embeddings.npy"
)

BERT_EMBEDDING_METADATA_PATH = (
    OUTPUTS_DIR / "bert_embedding_metadata.csv"
)


# ============================================================
# FEATURE RECONSTRUCTION OUTPUTS
# ============================================================

FEATURE_PREDICTIONS_PATH = (
    OUTPUTS_DIR / "feature_predictions.csv"
)

FEATURE_PREDICTION_METRICS_PATH = (
    OUTPUTS_DIR / "feature_prediction_metrics.csv"
)


# ============================================================
# MULTI-TASK MODEL ARTIFACTS
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


# ============================================================
# SENTIMENT FEATURES
# ============================================================

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

SENTIMENT_FEATURE_COUNT = len(
    SENTIMENT_FEATURE_COLUMNS
)


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

LDA_FEATURE_COUNT = len(
    LDA_FEATURE_COLUMNS
)


# ============================================================
# CHANNEL FEATURES
# ============================================================

CHANNEL_FEATURE_COLUMNS = [
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
]

CHANNEL_FEATURE_COUNT = len(
    CHANNEL_FEATURE_COLUMNS
)


# ============================================================
# STRUCTURAL / WEB FEATURES
# ============================================================

STRUCTURAL_FEATURE_COLUMNS = [
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
    "self_reference_min_shares",
    "self_reference_max_shares",
    "self_reference_avg_sharess",
]

STRUCTURAL_FEATURE_COUNT = len(
    STRUCTURAL_FEATURE_COLUMNS
)


# ============================================================
# FINAL MODEL FEATURE SPACE
# ============================================================

# Word features are intentionally absent.

MODEL_FEATURE_COLUMNS = (
    SENTIMENT_FEATURE_COLUMNS
    + LDA_FEATURE_COLUMNS
    + CHANNEL_FEATURE_COLUMNS
    + STRUCTURAL_FEATURE_COLUMNS
)

MODEL_FEATURE_COUNT = len(
    MODEL_FEATURE_COLUMNS
)


# ============================================================
# RECONSTRUCTED TEXT FEATURE SPACE
# ============================================================

RECONSTRUCTED_TEXT_FEATURE_COLUMNS = (
    SENTIMENT_FEATURE_COLUMNS
    + LDA_FEATURE_COLUMNS
)

RECONSTRUCTED_TEXT_FEATURE_COUNT = len(
    RECONSTRUCTED_TEXT_FEATURE_COLUMNS
)


# ============================================================
# DISTILBERT
# ============================================================

BERT_MODEL_NAME = "distilbert-base-uncased"

BERT_EMBEDDING_DIM = 768

BERT_MAX_LENGTH = 512

BERT_BATCH_SIZE = 8


# ============================================================
# MULTI-TASK FNN
# ============================================================

FNN_INPUT_DIM = BERT_EMBEDDING_DIM

FNN_SHARED_DIM_1 = 256

FNN_SHARED_DIM_2 = 128

FNN_SENTIMENT_OUTPUT_DIM = SENTIMENT_FEATURE_COUNT

FNN_LDA_OUTPUT_DIM = LDA_FEATURE_COUNT

FNN_DROPOUT = 0.30


# ============================================================
# FNN TRAINING
# ============================================================

FNN_BATCH_SIZE = 32

FNN_LEARNING_RATE = 1e-3

FNN_WEIGHT_DECAY = 1e-5

FNN_MAX_EPOCHS = 200

FNN_EARLY_STOPPING_PATIENCE = 20

FNN_MIN_DELTA = 1e-5


# Relative weight of the LDA loss.
#
# Total loss:
#
# sentiment_loss + LDA_LOSS_WEIGHT * lda_loss

LDA_LOSS_WEIGHT = 1.0


# ============================================================
# TARGET SCALING
# ============================================================

NORMALIZE_FEATURE_TARGETS = True


# ============================================================
# FEATURE EVALUATION
# ============================================================

FEATURE_METRICS = [
    "MAE",
    "RMSE",
    "R2",
    "Correlation",
]


# ============================================================
# DATA SPLITTING
# ============================================================

TRAIN_RATIO = 0.80

VALIDATION_RATIO = 0.10

TEST_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# SCRAPING
# ============================================================

SCRAPE_SAMPLE_SIZE = 1000

SCRAPE_MIN_TEXT_LENGTH = 200

SCRAPE_TIMEOUT = 15

SCRAPE_DELAY = 1.0

SCRAPE_RETRIES = 2

USE_WAYBACK_FALLBACK = True

WAYBACK_TIMEOUT = 20

WAYBACK_RETRIES = 2


# ============================================================
# REQUIRED ORIGINAL DATA COLUMNS
# ============================================================

# The six removed word features are deliberately absent.

REQUIRED_RAW_COLUMNS = [
    # Sentiment
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

    # LDA
    "LDA_00",
    "LDA_01",
    "LDA_02",
    "LDA_03",
    "LDA_04",

    # Channel
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",

    # Structural / web
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
    "self_reference_min_shares",
    "self_reference_max_shares",
    "self_reference_avg_sharess",

    # Target
    "shares",
]


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
# POPULARITY CLASSES
# ============================================================

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


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config():
    """Validate configuration consistency."""

    assert abs(
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
        - 1.0
    ) < 1e-9

    assert SENTIMENT_FEATURE_COUNT == 16

    assert LDA_FEATURE_COUNT == 5

    assert RECONSTRUCTED_TEXT_FEATURE_COUNT == 21

    assert CHANNEL_FEATURE_COUNT == 6

    assert FNN_INPUT_DIM == 768

    assert (
        FNN_SENTIMENT_OUTPUT_DIM
        == SENTIMENT_FEATURE_COUNT
    )

    assert (
        FNN_LDA_OUTPUT_DIM
        == LDA_FEATURE_COUNT
    )

    assert MODEL_FEATURE_COUNT == (
        SENTIMENT_FEATURE_COUNT
        + LDA_FEATURE_COUNT
        + CHANNEL_FEATURE_COUNT
        + STRUCTURAL_FEATURE_COUNT
    )

    assert FNN_BATCH_SIZE > 0

    assert FNN_MAX_EPOCHS > 0

    assert FNN_LEARNING_RATE > 0

    assert 0.0 <= FNN_DROPOUT < 1.0

    assert LDA_LOSS_WEIGHT >= 0.0


validate_config()