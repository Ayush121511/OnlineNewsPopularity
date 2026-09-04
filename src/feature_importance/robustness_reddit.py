"""
Method-robustness check -- Reddit domain, full 15,998 posts.
Same RF (max_depth=18 capped, per shap_reddit.py), same split, both
tasks x both variants. Adds group permutation importance alongside
SHAP.
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
import config  # noqa: E402
import data_loader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from permutation_robustness import group_permutation_importance  # noqa: E402

RANDOM_SEED = config.RANDOM_SEED
BERT_DIM = 768

FEATURE_GROUPS_POPULARITY = {
    "temporal": ["hour_of_day", "day_of_week", "is_weekend"],
    "source": ["domain_is_self", "domain_is_imgur", "domain_is_youtube", "subreddit_code"],
    "structure": ["n_tokens_title", "n_tokens_selftext", "has_selftext", "is_self", "has_thumbnail", "num_comments"],
    "meta": ["over_18", "title_has_question", "title_has_exclaim", "title_num_caps_words"],
}
FEATURE_GROUPS_TOPIC = {
    "temporal": FEATURE_GROUPS_POPULARITY["temporal"],
    "source": ["domain_is_self", "domain_is_imgur", "domain_is_youtube"],
    "structure": FEATURE_GROUPS_POPULARITY["structure"],
    "meta": FEATURE_GROUPS_POPULARITY["meta"],
}


def load_bert(dataset):
    meta = pd.read_csv(config.BERT_EMBEDDING_METADATA_PATH)
    emb = np.load(config.BERT_EMBEDDINGS_PATH)
    id_to_row = {row_id: i for i, row_id in enumerate(meta["id"])}
    idx = dataset["id"].map(id_to_row).to_numpy()
    return emb[idx]


def run_one(X, y, feature_cols, groups, label, n_repeats=5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=18, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    perm_rows = group_permutation_importance(clf, X_test, y_test, feature_cols, groups, n_repeats=n_repeats, seed=RANDOM_SEED)
    perm_df = pd.DataFrame(perm_rows)
    perm_df["pct_of_total_drop"] = 100.0 * perm_df["mean_acc_drop"].clip(lower=0) / perm_df["mean_acc_drop"].clip(lower=0).sum()

    print(f"\n--- {label} ---")
    print(f"Train: {len(y_train)}  Test: {len(y_test)}  Accuracy: {acc:.4f}")
    print(perm_df.sort_values("mean_acc_drop", ascending=False).to_string(index=False))
    return perm_df, acc


def main():
    print("=" * 70)
    print("PERMUTATION ROBUSTNESS -- REDDIT DOMAIN -- POPULARITY + TOPIC")
    print("=" * 70)

    dataset = data_loader.load_feature_prediction_dataset()
    print(f"Full dataset: {len(dataset)} posts, {dataset['subreddit'].nunique()} subreddits")

    out_dir = config.OUTPUTS_DIR / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    bert_cols = [f"bert_{i}" for i in range(BERT_DIM)]

    # ---------------- POPULARITY ----------------
    pop_cols = list(config.META_FEATURE_COLUMNS)
    X_tab = dataset[pop_cols].to_numpy(dtype=np.float64)
    y_pop = dataset["popularity_class"].to_numpy(dtype=np.int64)
    X_bert = load_bert(dataset)

    pop_tab_df, pop_tab_acc = run_one(X_tab, y_pop, pop_cols, FEATURE_GROUPS_POPULARITY, "POPULARITY tabular-only")

    groups_bert_pop = dict(FEATURE_GROUPS_POPULARITY)
    groups_bert_pop["bert"] = bert_cols
    X_combined = np.hstack([X_tab, X_bert])
    pop_bert_df, pop_bert_acc = run_one(X_combined, y_pop, pop_cols + bert_cols, groups_bert_pop, "POPULARITY tabular+BERT")

    pop_tab_df.to_csv(out_dir / "perm_popularity_reddit_full_tabular_only.csv", index=False)
    pop_bert_df.to_csv(out_dir / "perm_popularity_reddit_full_with_bert.csv", index=False)

    # ---------------- TOPIC (subreddit) ----------------
    topic_cols = [c for group in FEATURE_GROUPS_TOPIC.values() for c in group]
    le = LabelEncoder()
    y_topic = le.fit_transform(dataset["subreddit"])
    n_classes = len(le.classes_)

    X_tab_topic = dataset[topic_cols].to_numpy(dtype=np.float64)
    topic_tab_df, topic_tab_acc = run_one(X_tab_topic, y_topic, topic_cols, FEATURE_GROUPS_TOPIC, "TOPIC tabular-only")

    groups_bert_topic = dict(FEATURE_GROUPS_TOPIC)
    groups_bert_topic["bert"] = bert_cols
    X_combined_topic = np.hstack([X_tab_topic, X_bert])
    topic_bert_df, topic_bert_acc = run_one(X_combined_topic, y_topic, topic_cols + bert_cols, groups_bert_topic, "TOPIC tabular+BERT")

    topic_tab_df.to_csv(out_dir / "perm_topic_reddit_full_tabular_only.csv", index=False)
    topic_bert_df.to_csv(out_dir / "perm_topic_reddit_full_with_bert.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (permutation robustness, Reddit 15,998 posts)")
    print(f"Popularity: tabular={pop_tab_acc:.4f}  tabular+bert={pop_bert_acc:.4f}  chance=0.2500")
    print(f"Topic:      tabular={topic_tab_acc:.4f}  tabular+bert={topic_bert_acc:.4f}  chance={1/n_classes:.4f}")
    print("=" * 70)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
