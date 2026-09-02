# src/bert_embeddings.py

"""
Frozen DistilBERT embedding extraction.

This module:
    1. Loads the matched scraped article dataset.
    2. Loads frozen DistilBERT.
    3. Extracts one 768-dimensional embedding per article.
    4. Saves embeddings and article-ID metadata.

DistilBERT is used ONLY as a frozen feature extractor.
No BERT parameters are trained.
"""

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from transformers import (
    AutoModel,
    AutoTokenizer,
)

from config import (
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    BERT_BATCH_SIZE,
    BERT_EMBEDDING_DIM,
    BERT_EMBEDDINGS_PATH,
    BERT_EMBEDDING_METADATA_PATH,
)

from data_loader import (
    load_feature_prediction_dataset,
)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """
    Select the best available device.

    Priority:
        MPS -> CUDA -> CPU
    """

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
    Dataset containing article IDs and raw article text.

    We use title + article text as the transformer input.
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
# MODEL LOADING
# ============================================================

def load_model_and_tokenizer(device):
    """
    Load DistilBERT tokenizer and model.

    All model parameters are frozen.
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

    # --------------------------------------------------------
    # Freeze BERT
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return tokenizer, model


# ============================================================
# COLLATE FUNCTION
# ============================================================

def make_collate_fn(tokenizer):
    """
    Create a batch collation function using the tokenizer.
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
# EMBEDDING EXTRACTION
# ============================================================

def extract_embeddings(
    dataset,
    tokenizer,
    model,
    device,
):
    """
    Extract one 768-dimensional embedding per article.

    We use the first token representation from DistilBERT,
    corresponding to the leading special token representation.

    Returns:
        embeddings:
            numpy array with shape (N, 768)

        ids:
            numpy array containing article IDs in exactly the
            same order as the embeddings.
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

            # First-token representation.
            batch_embeddings = (
                outputs.last_hidden_state[
                    :,
                    0,
                    :
                ]
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
                batch_number
                * BERT_BATCH_SIZE,
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
    """
    Validate embedding dimensions and article-ID alignment.
    """

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, "
            f"got shape {embeddings.shape}"
        )

    if embeddings.shape[1] != BERT_EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dimension "
            f"{BERT_EMBEDDING_DIM}, "
            f"got {embeddings.shape[1]}"
        )

    # --------------------------------------------------------
    # Number of rows
    # --------------------------------------------------------

    if len(embeddings) != len(dataset):
        raise ValueError(
            "Number of embeddings does not match "
            "number of dataset articles."
        )

    if len(ids) != len(dataset):
        raise ValueError(
            "Number of embedding IDs does not match "
            "number of dataset articles."
        )

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    dataset_ids = dataset[
        "id"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        ids,
        dataset_ids,
    ):
        raise ValueError(
            "Embedding IDs are not aligned with "
            "the canonical dataset order."
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
    """
    Save embeddings and their article IDs.
    """

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
    print("FROZEN DISTILBERT EMBEDDING EXTRACTION")
    print("=" * 80)

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load canonical dataset
    # --------------------------------------------------------

    dataset = load_feature_prediction_dataset()

    print(
        f"Articles loaded: "
        f"{len(dataset):,}"
    )

    # --------------------------------------------------------
    # Load frozen model
    # --------------------------------------------------------

    tokenizer, model = (
        load_model_and_tokenizer(
            device
        )
    )

    # --------------------------------------------------------
    # Extract
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
        f"Number of articles: "
        f"{len(embeddings):,}"
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
        "\nFrozen BERT embedding check: PASSED"
    )


if __name__ == "__main__":
    main()