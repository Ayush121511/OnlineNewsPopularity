"""
Method-robustness check -- news domain, full 3,437 scrape.
Same RF, same split (random_seed) as shap_full_3437.py. Adds group
permutation importance alongside SHAP, both tasks x both variants.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config_news as config  # noqa: E402
import data_loader_news as data_loader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from permutation_robustness import group_permutation_importance  # noqa: E402

RANDOM_SEED = config.RANDOM_SEED
BERT_DIM = 768

TABULAR_GROUPS_POPULARITY = {
    "sentiment": config.SENTIMENT_FEATURE_COLUMNS,
    "lda": config.LDA_FEATURE_COLUMNS,
    "channel": config.CHANNEL_FEATURE_COLUMNS,
    "structural": config.STRUCTURAL_FEATURE_COLUMNS,
}
TABULAR_GROUPS_TOPIC = {
    "sentiment": config.SENTIMENT_FEATURE_COLUMNS,
    "lda": config.LDA_FEATURE_COLUMNS,
    "structural": config.STRUCTURAL_FEATURE_COLUMNS,
}
CHANNEL_COLS = config.CHANNEL_FEATURE_COLUMNS


def load_bert(dataset):
    meta = pd.read_csv(config.OUTPUTS_DIR / "feature_importance" / "news_full_bert_embedding_metadata.csv")
    emb = np.load(config.OUTPUTS_DIR / "feature_importance" / "news_full_bert_embeddings.npy")
    id_to_row = {row_id: i for i, row_id in enumerate(meta["id"])}
    idx = dataset["id"].map(id_to_row).to_numpy()
    return emb[idx]


def run_one(X, y, feature_cols, groups, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    perm_rows = group_permutation_importance(clf, X_test, y_test, feature_cols, groups, n_repeats=10, seed=RANDOM_SEED)
    perm_df = pd.DataFrame(perm_rows)
    perm_df["pct_of_total_drop"] = 100.0 * perm_df["mean_acc_drop"].clip(lower=0) / perm_df["mean_acc_drop"].clip(lower=0).sum()

    print(f"\n--- {label} ---")
    print(f"Test: {len(y_test)}  Accuracy: {acc:.4f}")
    print(perm_df.sort_values("mean_acc_drop", ascending=False).to_string(index=False))
    return perm_df, acc


def main():
    print("=" * 70)
    print("PERMUTATION ROBUSTNESS -- NEWS 3,437 -- POPULARITY + TOPIC")
    print("=" * 70)

    dataset = data_loader.load_feature_prediction_dataset()
    out_dir = config.OUTPUTS_DIR / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    bert_cols = [f"bert_{i}" for i in range(BERT_DIM)]

    # ---------------- POPULARITY ----------------
    X_bert = load_bert(dataset)
    tab_cols = config.MODEL_FEATURE_COLUMNS
    X_tab = dataset[tab_cols].to_numpy(dtype=np.float64)
    y_pop = dataset["popularity_class"].to_numpy(dtype=np.int64)

    pop_tab_df, pop_tab_acc = run_one(X_tab, y_pop, tab_cols, TABULAR_GROUPS_POPULARITY, "POPULARITY tabular-only")

    X_combined = np.hstack([X_tab, X_bert])
    groups_bert = dict(TABULAR_GROUPS_POPULARITY)
    groups_bert["bert"] = bert_cols
    pop_bert_df, pop_bert_acc = run_one(X_combined, y_pop, tab_cols + bert_cols, groups_bert, "POPULARITY tabular+BERT")

    pop_tab_df.to_csv(out_dir / "perm_popularity_news_3437_tabular_only.csv", index=False)
    pop_bert_df.to_csv(out_dir / "perm_popularity_news_3437_with_bert.csv", index=False)

    # ---------------- TOPIC ----------------
    onehot = dataset[CHANNEL_COLS]
    has_channel = onehot.sum(axis=1) == 1
    topic_dataset = dataset.loc[has_channel].copy()
    topic_dataset["topic"] = onehot.loc[has_channel].idxmax(axis=1).str.replace("data_channel_is_", "", regex=False)

    le = LabelEncoder()
    y_topic = le.fit_transform(topic_dataset["topic"])
    X_bert_topic = load_bert(topic_dataset)
    tab_cols_topic = config.SENTIMENT_FEATURE_COLUMNS + config.LDA_FEATURE_COLUMNS + config.STRUCTURAL_FEATURE_COLUMNS
    X_tab_topic = topic_dataset[tab_cols_topic].to_numpy(dtype=np.float64)

    topic_tab_df, topic_tab_acc = run_one(X_tab_topic, y_topic, tab_cols_topic, TABULAR_GROUPS_TOPIC, "TOPIC tabular-only")

    X_combined_topic = np.hstack([X_tab_topic, X_bert_topic])
    groups_bert_topic = dict(TABULAR_GROUPS_TOPIC)
    groups_bert_topic["bert"] = bert_cols
    topic_bert_df, topic_bert_acc = run_one(X_combined_topic, y_topic, tab_cols_topic + bert_cols, groups_bert_topic, "TOPIC tabular+BERT")

    topic_tab_df.to_csv(out_dir / "perm_topic_news_3437_tabular_only.csv", index=False)
    topic_bert_df.to_csv(out_dir / "perm_topic_news_3437_with_bert.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (permutation robustness, news 3437)")
    print(f"Popularity: tabular={pop_tab_acc:.4f}  tabular+bert={pop_bert_acc:.4f}")
    print(f"Topic:      tabular={topic_tab_acc:.4f}  tabular+bert={topic_bert_acc:.4f}")
    print("=" * 70)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
