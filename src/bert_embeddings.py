# src/bert_embeddings.py

"""
Generate frozen DistilBERT embeddings for article title + text.

Pipeline:
    canonical dataset
        ↓
    title + article text
        ↓
    DistilBERT tokenizer
        ↓
    frozen DistilBERT
        ↓
    masked mean pooling
        ↓
    768-d embedding

Output:
    outputs/bert_embeddings.npy
    outputs/bert_embedding_metadata.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# ------------------------------------------------------------------
# Project path
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import (
    BERT_MODEL_NAME,
    BERT_EMBEDDING_DIM,
    BERT_MAX_LENGTH,
    BERT_BATCH_SIZE,
    BERT_EMBEDDINGS_PATH,
    BERT_EMBEDDING_METADATA_PATH,
)

from data_loader import load_feature_prediction_dataset


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class ArticleDataset(Dataset):

    def __init__(self, titles, texts):
        self.titles = titles
        self.texts = texts

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, index):
        title = str(self.titles[index]).strip()
        text = str(self.texts[index]).strip()

        if title:
            combined_text = title + " [SEP] " + text
        else:
            combined_text = text

        return combined_text


# ------------------------------------------------------------------
# Mean pooling
# ------------------------------------------------------------------

def masked_mean_pooling(
    hidden_states,
    attention_mask,
):
    mask = attention_mask.unsqueeze(-1).expand(
        hidden_states.size()
    ).float()

    masked_embeddings = hidden_states * mask

    summed = masked_embeddings.sum(dim=1)

    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


# ------------------------------------------------------------------
# Embedding generation
# ------------------------------------------------------------------

def generate_embeddings(
    dataset,
    tokenizer,
    model,
    device,
):
    article_dataset = ArticleDataset(
        dataset["title"].tolist(),
        dataset["text"].tolist(),
    )

    loader = DataLoader(
        article_dataset,
        batch_size=BERT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    embeddings = []

    model.eval()

    with torch.no_grad():

        for batch_number, texts in enumerate(loader, start=1):

            encoded = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=BERT_MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded
            )

            pooled = masked_mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            embeddings.append(
                pooled.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if batch_number % 25 == 0:
                print(
                    f"Processed "
                    f"{batch_number * BERT_BATCH_SIZE:,} "
                    f"articles..."
                )

    embeddings = np.vstack(embeddings)

    return embeddings


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    print("=" * 70)
    print("BERT EMBEDDING GENERATION")
    print("=" * 70)

    device = get_device()

    print(f"\nModel  : {BERT_MODEL_NAME}")
    print(f"Device : {device}")
    print(f"Max len: {BERT_MAX_LENGTH}")
    print(f"Batch  : {BERT_BATCH_SIZE}")

    # --------------------------------------------------------------
    # Load canonical dataset
    # --------------------------------------------------------------

    dataset = load_feature_prediction_dataset()

    print(
        f"\nArticles: {len(dataset):,}"
    )

    # --------------------------------------------------------------
    # Load tokenizer + frozen model
    # --------------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    print("Loading model...")

    model = AutoModel.from_pretrained(
        BERT_MODEL_NAME
    )

    model.to(device)

    # Important: BERT is frozen.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    # --------------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------------

    expected_shape = (
        len(dataset),
        BERT_EMBEDDING_DIM,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape: "
            f"{embeddings.shape}; "
            f"expected {expected_shape}."
        )

    # --------------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------------

    BERT_EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        BERT_EMBEDDINGS_PATH,
        embeddings,
    )

    # --------------------------------------------------------------
    # Save metadata
    #
    # Keep row alignment explicit.
    # --------------------------------------------------------------

    metadata = dataset[
        [
            "id",
            "score",
            "popularity_class",
        ]
    ].copy()

    metadata.to_csv(
        BERT_EMBEDDING_METADATA_PATH,
        index=False,
    )

    # --------------------------------------------------------------
    # Final diagnostics
    # --------------------------------------------------------------

    print("\nSaved:")
    print(
        f"  Embeddings : {BERT_EMBEDDINGS_PATH}"
    )
    print(
        f"  Metadata   : "
        f"{BERT_EMBEDDING_METADATA_PATH}"
    )

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    print("\nBERT embedding generation: PASSED")


if __name__ == "__main__":
    main()