"""
Extract frozen embeddings from a sentiment-fine-tuned RoBERTa model.

Pipeline:
    title + article text
        ↓
    frozen RoBERTa sentiment encoder
        ↓
    masked mean pooling
        ↓
    768-dimensional embedding
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config import (
    SENTIMENT_BERT_MODEL_NAME,
    SENTIMENT_BERT_EMBEDDING_DIM,
    SENTIMENT_BERT_MAX_LENGTH,
    SENTIMENT_BERT_BATCH_SIZE,
    SENTIMENT_BERT_EMBEDDINGS_PATH,
    SENTIMENT_BERT_EMBEDDING_METADATA_PATH,
)

from data_loader import load_feature_prediction_dataset


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

dataset = load_feature_prediction_dataset()

titles = dataset["title"].fillna("").astype(str).tolist()
texts = dataset["text"].fillna("").astype(str).tolist()
ids = dataset["id"].to_numpy()


# ============================================================
# LOAD SENTIMENT MODEL
# ============================================================

print()
print("Loading sentiment-fine-tuned RoBERTa...")
print(f"Model: {SENTIMENT_BERT_MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    SENTIMENT_BERT_MODEL_NAME
)

sequence_model = (
    AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_BERT_MODEL_NAME
    )
)

sequence_model = sequence_model.to(DEVICE)
sequence_model.eval()


# ============================================================
# FREEZE MODEL
# ============================================================
# We use the sentiment-fine-tuned encoder as a frozen
# representation extractor. No Transformer parameters are
# updated during this experiment.

for parameter in sequence_model.parameters():
    parameter.requires_grad = False


# The actual RoBERTa encoder is the `.roberta` component.
encoder = sequence_model.roberta

encoder.eval()


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

all_embeddings = []

print()
print("Extracting sentiment-aware embeddings...")
print("=" * 60)

with torch.no_grad():

    for start in range(
        0,
        len(dataset),
        SENTIMENT_BERT_BATCH_SIZE,
    ):

        end = min(
            start + SENTIMENT_BERT_BATCH_SIZE,
            len(dataset),
        )

        batch_titles = titles[start:end]
        batch_texts = texts[start:end]

        combined_texts = [
            (
                title.strip()
                + " "
                + text.strip()
            ).strip()
            for title, text in zip(
                batch_titles,
                batch_texts,
            )
        ]

        encoded = tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=SENTIMENT_BERT_MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(DEVICE)

        attention_mask = encoded[
            "attention_mask"
        ].to(DEVICE)

        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        # ----------------------------------------------------
        # Masked mean pooling
        # ----------------------------------------------------

        mask = attention_mask.unsqueeze(-1).to(
            hidden_states.dtype
        )

        masked_hidden_states = (
            hidden_states * mask
        )

        summed_embeddings = (
            masked_hidden_states.sum(dim=1)
        )

        token_counts = (
            mask.sum(dim=1)
            .clamp(min=1e-9)
        )

        pooled_embeddings = (
            summed_embeddings / token_counts
        )

        all_embeddings.append(
            pooled_embeddings.cpu().numpy()
        )

        print(
            f"Processed {end:4d}/{len(dataset)}"
        )


# ============================================================
# COMBINE EMBEDDINGS
# ============================================================

embeddings = np.vstack(all_embeddings)


# ============================================================
# VALIDATION
# ============================================================

if embeddings.shape != (
    len(dataset),
    SENTIMENT_BERT_EMBEDDING_DIM,
):
    raise ValueError(
        "Unexpected embedding shape: "
        f"{embeddings.shape}"
    )


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

SENTIMENT_BERT_EMBEDDINGS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

np.save(
    SENTIMENT_BERT_EMBEDDINGS_PATH,
    embeddings,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = pd.DataFrame(
    {
        "id": ids,
        "embedding_index": np.arange(
            len(dataset)
        ),
    }
)

metadata.to_csv(
    SENTIMENT_BERT_EMBEDDING_METADATA_PATH,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("SENTIMENT BERT EMBEDDING EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Model:       {SENTIMENT_BERT_MODEL_NAME}"
)

print(
    f"Embeddings:  {embeddings.shape}"
)

print(
    f"Saved to:    {SENTIMENT_BERT_EMBEDDINGS_PATH}"
)

print(
    f"Metadata:    {SENTIMENT_BERT_EMBEDDING_METADATA_PATH}"
)