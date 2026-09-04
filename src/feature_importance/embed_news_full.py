"""
Generate DistilBERT embeddings for the FULL current news scrape
(3,437 articles, data/scraped_articles.csv working-tree version --
larger than the 996-row git-HEAD-pinned set used earlier).

Frozen DistilBERT, masked mean pooling, same method as
src/bert_embeddings.py (HEAD version). Output kept separate from
outputs/bert_embeddings.npy, which currently holds the Reddit-domain
embeddings -- do not overwrite that.

Output:
    outputs/feature_importance/news_full_bert_embeddings.npy
    outputs/feature_importance/news_full_bert_embedding_metadata.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config_news as config  # noqa: E402
import data_loader_news as data_loader  # noqa: E402

OUT_DIR = config.OUTPUTS_DIR / "feature_importance"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_PATH = OUT_DIR / "news_full_bert_embeddings.npy"
META_PATH = OUT_DIR / "news_full_bert_embedding_metadata.csv"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ArticleTextDataset(Dataset):
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
        return {"id": int(self.ids[index]), "text": self.texts[index]}


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        ids = [item["id"] for item in batch]
        texts = [item["text"] for item in batch]
        encoded = tokenizer(
            texts, padding=True, truncation=True,
            max_length=config.BERT_MAX_LENGTH, return_tensors="pt",
        )
        return {"ids": ids, "input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
    return collate_fn


def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def main():
    print("=" * 70)
    print("BERT EMBEDDING GENERATION -- FULL NEWS SCRAPE (3,437)")
    print("=" * 70)

    dataset = data_loader.load_feature_prediction_dataset()
    print(f"Matched dataset: {len(dataset)} articles (scraped x UCI raw, inner join)")

    device = get_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    article_dataset = ArticleTextDataset(dataset)
    dataloader = DataLoader(
        article_dataset, batch_size=config.BERT_BATCH_SIZE,
        shuffle=False, collate_fn=make_collate_fn(tokenizer),
    )

    all_embeddings, all_ids = [], []
    total = len(article_dataset)
    with torch.no_grad():
        for batch_number, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_emb = mean_pooling(outputs.last_hidden_state, attention_mask).detach().cpu().numpy()
            all_embeddings.append(batch_emb)
            all_ids.extend(batch["ids"])
            processed = min(batch_number * config.BERT_BATCH_SIZE, total)
            if batch_number % 20 == 0 or processed == total:
                print(f"Processed: {processed:,}/{total:,}")

    embeddings = np.concatenate(all_embeddings, axis=0)
    ids = np.asarray(all_ids, dtype=np.int64)

    np.save(EMB_PATH, embeddings)
    pd.DataFrame({"id": ids}).to_csv(META_PATH, index=False)

    print(f"\nSaved: {EMB_PATH}  shape={embeddings.shape}")
    print(f"Saved: {META_PATH}")


if __name__ == "__main__":
    main()
