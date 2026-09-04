"""
SHAP feature-group importance -- Reddit domain, both tasks
(popularity, topic=subreddit), tabular-only vs tabular+BERT.

Uses live src/config.py + src/data_loader.py directly (both are
correctly Reddit-configured in the working tree, no HEAD-pinning
needed here) and outputs/bert_embeddings.npy (already the Reddit
embeddings, 15,998 x 768, id-aligned via bert_embedding_metadata.csv).

Feature groups (17 meta cols total):
    temporal:   hour_of_day, day_of_week, is_weekend
    source:     domain_is_self, domain_is_imgur, domain_is_youtube, subreddit_code
    structure:  n_tokens_title, n_tokens_selftext, has_selftext, is_self,
                has_thumbnail, num_comments
    meta:       over_18, title_has_question, title_has_exclaim, title_num_caps_words

subreddit_code is dropped for the topic task (it's ~ the label).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config  # noqa: E402
import data_loader  # noqa: E402

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
    "source": ["domain_is_self", "domain_is_imgur", "domain_is_youtube"],  # subreddit_code dropped -- ~ label
    "structure": FEATURE_GROUPS_POPULARITY["structure"],
    "meta": FEATURE_GROUPS_POPULARITY["meta"],
}


def mean_abs_shap_per_feature(shap_values, n_features):
    if isinstance(shap_values, list):
        return np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    shap_arr = np.asarray(shap_values)
    feature_axis = [ax for ax, size in enumerate(shap_arr.shape) if size == n_features][0]
    other_axes = tuple(ax for ax in range(shap_arr.ndim) if ax != feature_axis)
    return np.abs(shap_arr).mean(axis=other_axes)


def run_one(X, y, feature_cols, groups, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=18, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    per_feature = mean_abs_shap_per_feature(shap_values, len(feature_cols))
    importance_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": per_feature})

    rows = []
    for group_name, cols in groups.items():
        vals = importance_df[importance_df["feature"].isin(cols)]["mean_abs_shap"]
        rows.append({
            "group": group_name,
            "n_features": len(cols),
            "total_mean_abs_shap": vals.sum(),
            "avg_mean_abs_shap_per_feature": vals.mean(),
        })
    group_df = pd.DataFrame(rows)
    group_df["pct_of_total"] = 100.0 * group_df["total_mean_abs_shap"] / group_df["total_mean_abs_shap"].sum()
    group_df["pct_of_avg_per_feature"] = 100.0 * group_df["avg_mean_abs_shap_per_feature"] / group_df["avg_mean_abs_shap_per_feature"].sum()

    print(f"\n--- {label} ---")
    print(f"Train: {len(y_train)}  Test: {len(y_test)}  Accuracy: {acc:.4f}")
    print(group_df.sort_values("avg_mean_abs_shap_per_feature", ascending=False).to_string(index=False))
    return group_df, acc


def load_bert(dataset):
    meta = pd.read_csv(config.BERT_EMBEDDING_METADATA_PATH)
    emb = np.load(config.BERT_EMBEDDINGS_PATH)
    id_to_row = {row_id: i for i, row_id in enumerate(meta["id"])}
    idx = dataset["id"].map(id_to_row).to_numpy()
    return emb[idx]


def main():
    print("=" * 70)
    print("SHAP -- REDDIT DOMAIN -- POPULARITY + TOPIC")
    print("=" * 70)

    dataset = data_loader.load_feature_prediction_dataset()
    print(f"Full dataset: {len(dataset)} posts, {dataset['subreddit'].nunique()} subreddits")

    full_run = "--full" in sys.argv
    if not full_run:
        SANITY_SIZE = 1000
        dataset, _ = train_test_split(
            dataset, train_size=SANITY_SIZE,
            stratify=dataset["popularity_class"], random_state=RANDOM_SEED,
        )
        print(f"Sanity subset: {len(dataset)} posts")

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

    tag = "full" if full_run else "sanity"
    pop_tab_df.to_csv(out_dir / f"shap_popularity_reddit_{tag}_tabular_only.csv", index=False)
    pop_bert_df.to_csv(out_dir / f"shap_popularity_reddit_{tag}_with_bert.csv", index=False)

    # ---------------- TOPIC (subreddit) ----------------
    topic_cols = [c for group in FEATURE_GROUPS_TOPIC.values() for c in group]
    le = LabelEncoder()
    y_topic = le.fit_transform(dataset["subreddit"])
    n_classes = len(le.classes_)
    majority = dataset["subreddit"].value_counts(normalize=True).max()
    print(f"\nTopic classes: {n_classes}  chance={1/n_classes:.4f}  majority_baseline={majority:.4f}")

    X_tab_topic = dataset[topic_cols].to_numpy(dtype=np.float64)
    topic_tab_df, topic_tab_acc = run_one(X_tab_topic, y_topic, topic_cols, FEATURE_GROUPS_TOPIC, "TOPIC tabular-only")

    groups_bert_topic = dict(FEATURE_GROUPS_TOPIC)
    groups_bert_topic["bert"] = bert_cols
    X_combined_topic = np.hstack([X_tab_topic, X_bert])
    topic_bert_df, topic_bert_acc = run_one(X_combined_topic, y_topic, topic_cols + bert_cols, groups_bert_topic, "TOPIC tabular+BERT")

    topic_tab_df.to_csv(out_dir / f"shap_topic_reddit_{tag}_tabular_only.csv", index=False)
    topic_bert_df.to_csv(out_dir / f"shap_topic_reddit_{tag}_with_bert.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (Reddit, 15,998 posts)")
    print(f"Popularity: tabular={pop_tab_acc:.4f}  tabular+bert={pop_bert_acc:.4f}  chance=0.2500")
    print(f"Topic:      tabular={topic_tab_acc:.4f}  tabular+bert={topic_bert_acc:.4f}  chance={1/n_classes:.4f}  majority={majority:.4f}")
    print("=" * 70)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
