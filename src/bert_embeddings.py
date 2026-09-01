"""
bert_embeddings.py

Extract frozen DistilBERT embeddings from scraped article text.

Pipeline:

    scraped + matched articles
              ↓
        DistilBERT tokenizer
              ↓
       Frozen DistilBERT
              ↓
        768-d embeddings
              ↓
        saved to disk

The embeddings are later used as input to feed-forward neural networks.

No DistilBERT parameters are trained in this module.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from config import (
    TRANSFORMER_MODEL_NAME,
    TRANSFORMER_MAX_LENGTH,
    TRANSFORMER_BATCH_SIZE,
    RANDOM_SEED,
    OUTPUT_DIR,
)
from data_loader import load_feature_prediction_dataset


# =============================================================================
# OUTPUT PATHS
# =============================================================================

EMBEDDINGS_PATH = OUTPUT_DIR / "bert_embeddings.npy"
EMBEDDING_METADATA_PATH = OUTPUT_DIR / "bert_embedding_metadata.csv"


# =============================================================================
# DEVICE
# =============================================================================

def get_device() -> torch.device:
    """
    Select the best available device.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =============================================================================
# DATASET
# =============================================================================

class ArticleTextDataset(Dataset):
    """
    PyTorch dataset containing article text.

    The dataset returns only text and article ID.
    Target features are deliberately not passed to the embedding model.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):
        self.dataframe = dataframe.reset_index(
            drop=True
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]

        return {
            "id": int(row["id"]),
            "text": str(row["text"]),
        }


# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def create_collate_fn(
    tokenizer,
):
    """
    Create a batch collation function using the configured tokenizer.
    """

    def collate_fn(batch):

        ids = [
            item["id"]
            for item in batch
        ]

        texts = [
            item["text"]
            for item in batch
        ]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=TRANSFORMER_MAX_LENGTH,
            return_tensors="pt",
        )

        encoded["article_ids"] = ids

        return encoded

    return collate_fn


# =============================================================================
# MODEL
# =============================================================================

def load_embedding_model(
    device: torch.device,
):
    """
    Load the pretrained transformer and tokenizer.

    The transformer is completely frozen.
    """

    print(
        f"Loading transformer: "
        f"{TRANSFORMER_MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TRANSFORMER_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        TRANSFORMER_MODEL_NAME
    )

    # -------------------------------------------------------------------------
    # Freeze every transformer parameter.
    # -------------------------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    model.to(device)

    return tokenizer, model


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_embeddings(
    dataframe: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract one fixed-size embedding for every article.

    We use the first-token ([CLS]) representation.

    Returns
    -------
    embeddings:
        Array of shape:

            (number_of_articles, hidden_size)

    article_ids:
        IDs corresponding exactly to the embedding rows.
    """

    dataset = ArticleTextDataset(
        dataframe
    )

    collate_fn = create_collate_fn(
        tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=TRANSFORMER_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_embeddings = []
    all_ids = []

    print(
        f"Extracting embeddings for "
        f"{len(dataset):,} articles..."
    )

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):

        article_ids = batch.pop(
            "article_ids"
        )

        input_ids = batch["input_ids"].to(
            device
        )

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        # -------------------------------------------------------------
        # Frozen transformer forward pass
        # -------------------------------------------------------------

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # -------------------------------------------------------------
        # First-token representation
        #
        # DistilBERT has no pooled_output.
        # The representation of the first token is used as the
        # fixed article embedding.
        # -------------------------------------------------------------

        embeddings = outputs.last_hidden_state[
            :,
            0,
            :,
        ]

        # Move to CPU immediately so GPU/MPS memory stays low.
        embeddings = embeddings.cpu().numpy()

        all_embeddings.append(
            embeddings
        )

        all_ids.extend(
            article_ids
        )

        print(
            f"  Batch "
            f"{batch_number}/{len(loader)}"
        )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    return embeddings, all_ids


# =============================================================================
# SAVE EMBEDDINGS
# =============================================================================

def save_embeddings(
    embeddings: np.ndarray,
    article_ids: list[int],
    dataframe: pd.DataFrame,
) -> None:
    """
    Save embeddings and their metadata.

    Metadata preserves the mapping:

        embedding row ↔ article ID ↔ URL

    This is essential because the embeddings and target features must
    never become misaligned.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    metadata = dataframe[
        [
            "id",
            "url",
        ]
    ].copy()

    # Reorder according to the embedding IDs.
    id_to_row = {
        int(article_id): index
        for index, article_id
        in enumerate(article_ids)
    }

    metadata["_embedding_row"] = (
        metadata["id"]
        .astype(int)
        .map(id_to_row)
    )

    metadata = metadata.sort_values(
        "_embedding_row"
    )

    metadata.to_csv(
        EMBEDDING_METADATA_PATH,
        index=False,
    )

    print()
    print(
        f"Embeddings saved to:\n"
        f"{EMBEDDINGS_PATH}"
    )

    print(
        f"Metadata saved to:\n"
        f"{EMBEDDING_METADATA_PATH}"
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_embeddings(
    embeddings: np.ndarray,
    article_ids: list[int],
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate embedding dimensions and article alignment.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding array, "
            f"got shape {embeddings.shape}."
        )

    if embeddings.shape[0] != len(dataframe):
        raise ValueError(
            "Number of embeddings does not match "
            "number of articles."
        )

    if embeddings.shape[0] != len(article_ids):
        raise ValueError(
            "Number of embedding IDs does not match "
            "number of embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    dataframe_ids = (
        dataframe["id"]
        .astype(int)
        .tolist()
    )

    if dataframe_ids != article_ids:
        raise ValueError(
            "Embedding article IDs are not aligned "
            "with the input dataframe."
        )


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def run_embedding_extraction() -> None:
    """
    Execute the complete frozen-transformer embedding pipeline.
    """

    print("=" * 70)
    print("FROZEN DISTILBERT EMBEDDING EXTRACTION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------

    torch.manual_seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    # -------------------------------------------------------------------------
    # Load matched text + target dataset
    # -------------------------------------------------------------------------

    dataframe = (
        load_feature_prediction_dataset()
    )

    print(
        f"Articles available: "
        f"{len(dataframe):,}"
    )

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = get_device()

    print(
        f"Device: {device}"
    )

    # -------------------------------------------------------------------------
    # Load frozen transformer
    # -------------------------------------------------------------------------

    tokenizer, model = load_embedding_model(
        device
    )

    # -------------------------------------------------------------------------
    # Extract embeddings
    # -------------------------------------------------------------------------

    embeddings, article_ids = (
        extract_embeddings(
            dataframe=dataframe,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_embeddings(
        embeddings=embeddings,
        article_ids=article_ids,
        dataframe=dataframe,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_embeddings(
        embeddings=embeddings,
        article_ids=article_ids,
        dataframe=dataframe,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Number of articles : "
        f"{embeddings.shape[0]:,}"
    )

    print(
        f"Embedding dimension : "
        f"{embeddings.shape[1]:,}"
    )

    print(
        f"Embedding shape     : "
        f"{embeddings.shape}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_embedding_extraction()