# src/bert_embeddings.py

"""
Frozen DistilBERT embedding extraction for the LDA-only experiment.

Pipeline
--------
Article title + text
        ↓
Frozen DistilBERT
        ↓
masked mean pooling
        ↓
768-dimensional embedding
        ↓
saved for LDA feature prediction

DistilBERT is never fine-tuned.
"""

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer

from config import (
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    BERT_BATCH_SIZE,
    BERT_EMBEDDING_DIM,
    BERT_EMBEDDINGS_PATH,
    BERT_EMBEDDING_METADATA_PATH,
)

from data_loader import load_feature_prediction_dataset


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """Select the best available PyTorch device."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

class ArticleTextDataset(Dataset):
    """
    Dataset containing article IDs and article text.

    Title and article body are concatenated because both are
    part of the textual information available for reconstruction.
    """

    def __init__(self, dataframe):
        self.ids = dataframe["id"].to_numpy()

        self.texts = (
            dataframe["title"].fillna("").astype(str)
            + " "
            + dataframe["text"].fillna("").astype(str)
        ).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {
            "id": int(self.ids[index]),
            "text": self.texts[index],
        }


# ============================================================
# MODEL
# ============================================================

def load_model_and_tokenizer(device):
    """
    Load DistilBERT and tokenizer.

    All DistilBERT parameters are frozen.
    """

    print(
        f"Loading model: {BERT_MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        BERT_MODEL_NAME
    )

    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return tokenizer, model


# ============================================================
# COLLATE
# ============================================================

def make_collate_fn(tokenizer):
    """Create tokenizer-based batch collation."""

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
            max_length=BERT_MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "ids": ids,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    return collate_fn


# ============================================================
# MASKED MEAN POOLING
# ============================================================

def mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked mean pooling over token embeddings.

    Args:
        hidden_states:
            Shape:
                (batch_size, sequence_length, hidden_dim)

        attention_mask:
            Shape:
                (batch_size, sequence_length)

    Returns:
        Pooled embeddings:
            (batch_size, hidden_dim)
    """

    # Convert mask from:
    #   (batch_size, sequence_length)
    # to:
    #   (batch_size, sequence_length, 1)
    mask = attention_mask.unsqueeze(-1).to(
        hidden_states.dtype
    )

    # Zero out padding-token embeddings.
    masked_hidden_states = (
        hidden_states * mask
    )

    # Sum valid token embeddings.
    summed_embeddings = (
        masked_hidden_states.sum(dim=1)
    )

    # Number of valid tokens for each article.
    token_counts = (
        mask.sum(dim=1).clamp(min=1e-9)
    )

    # Mean over valid tokens only.
    pooled_embeddings = (
        summed_embeddings
        / token_counts
    )

    return pooled_embeddings


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

def extract_embeddings(
    dataset,
    tokenizer,
    model,
    device,
):
    """
    Extract one 768-dimensional mean-pooled embedding per article.

    Returns:
        embeddings:
            Shape (N, 768)

        ids:
            Article IDs in exactly the same order.
    """

    article_dataset = ArticleTextDataset(
        dataset
    )

    dataloader = DataLoader(
        article_dataset,
        batch_size=BERT_BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(
            tokenizer
        ),
    )

    all_embeddings = []
    all_ids = []

    total = len(article_dataset)

    with torch.no_grad():

        for batch_number, batch in enumerate(
            dataloader,
            start=1,
        ):

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # ------------------------------------------------
            # Masked mean pooling
            # ------------------------------------------------

            batch_embeddings = mean_pooling(
                outputs.last_hidden_state,
                attention_mask,
            )

            batch_embeddings = (
                batch_embeddings
                .detach()
                .cpu()
                .numpy()
            )

            all_embeddings.append(
                batch_embeddings
            )

            all_ids.extend(
                batch["ids"]
            )

            processed = min(
                batch_number * BERT_BATCH_SIZE,
                total,
            )

            print(
                f"Processed embeddings: "
                f"{processed:,}/{total:,}"
            )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    ids = np.asarray(
        all_ids,
        dtype=np.int64,
    )

    return embeddings, ids


# ============================================================
# VALIDATION
# ============================================================

def validate_embeddings(
    embeddings,
    ids,
    dataset,
):
    """Validate embedding shape, values, and ID alignment."""

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, "
            f"got {embeddings.shape}"
        )

    if embeddings.shape[1] != BERT_EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dimension "
            f"{BERT_EMBEDDING_DIM}, "
            f"got {embeddings.shape[1]}"
        )

    if len(embeddings) != len(dataset):
        raise ValueError(
            "Embedding count does not match "
            "dataset article count."
        )

    if len(ids) != len(dataset):
        raise ValueError(
            "Embedding ID count does not match "
            "dataset article count."
        )

    dataset_ids = dataset[
        "id"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        ids,
        dataset_ids,
    ):
        raise ValueError(
            "Embedding IDs are not aligned with "
            "the canonical dataset ordering."
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )


# ============================================================
# SAVE
# ============================================================

def save_embeddings(
    embeddings,
    ids,
):
    """Save embeddings and article-ID metadata."""

    BERT_EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        BERT_EMBEDDINGS_PATH,
        embeddings,
    )

    metadata = pd.DataFrame({
        "id": ids,
    })

    metadata.to_csv(
        BERT_EMBEDDING_METADATA_PATH,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FROZEN DISTILBERT EMBEDDINGS — LDA ONLY")
    print("=" * 80)

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    dataset = load_feature_prediction_dataset()

    print(
        f"Articles loaded: "
        f"{len(dataset):,}"
    )

    # --------------------------------------------------------
    # Load frozen BERT
    # --------------------------------------------------------

    tokenizer, model = (
        load_model_and_tokenizer(
            device
        )
    )

    # --------------------------------------------------------
    # Extract mean-pooled embeddings
    # --------------------------------------------------------

    embeddings, ids = extract_embeddings(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_embeddings(
        embeddings=embeddings,
        ids=ids,
        dataset=dataset,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_embeddings(
        embeddings,
        ids,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"\nEmbeddings saved to:\n"
        f"{BERT_EMBEDDINGS_PATH}"
    )

    print(
        f"\nMetadata saved to:\n"
        f"{BERT_EMBEDDING_METADATA_PATH}"
    )

    print(
        "\nMean-pooling embedding check: PASSED"
    )


if __name__ == "__main__":
    main()