"""
SHAP feature-group importance on the REAL full news scrape (3,437
articles) -- both tasks: popularity and topic classification, each
tabular-only vs tabular+BERT. Uses the freshly generated
news_full_bert_embeddings.npy (embed_news_full.py), not the 996-row
HEAD-pinned set used earlier.
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
import config_news as config  # noqa: E402
import data_loader_news as data_loader  # noqa: E402

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
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
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
    meta = pd.read_csv(config.OUTPUTS_DIR / "feature_importance" / "news_full_bert_embedding_metadata.csv")
    emb = np.load(config.OUTPUTS_DIR / "feature_importance" / "news_full_bert_embeddings.npy")
    id_to_row = {row_id: i for i, row_id in enumerate(meta["id"])}
    idx = dataset["id"].map(id_to_row).to_numpy()
    return emb[idx]


def main():
    print("=" * 70)
    print("SHAP -- FULL 3,437 NEWS SCRAPE -- POPULARITY + TOPIC")
    print("=" * 70)

    dataset = data_loader.load_feature_prediction_dataset()
    print(f"Popularity dataset: {len(dataset)} articles")

    out_dir = config.OUTPUTS_DIR / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- POPULARITY ----------------
    X_bert = load_bert(dataset)
    tab_cols = config.MODEL_FEATURE_COLUMNS
    X_tab = dataset[tab_cols].to_numpy(dtype=np.float64)
    y_pop = dataset["popularity_class"].to_numpy(dtype=np.int64)

    pop_tab_df, pop_tab_acc = run_one(X_tab, y_pop, tab_cols, TABULAR_GROUPS_POPULARITY, "POPULARITY tabular-only")

    bert_cols = [f"bert_{i}" for i in range(BERT_DIM)]
    X_combined = np.hstack([X_tab, X_bert])
    groups_bert = dict(TABULAR_GROUPS_POPULARITY)
    groups_bert["bert"] = bert_cols
    pop_bert_df, pop_bert_acc = run_one(X_combined, y_pop, tab_cols + bert_cols, groups_bert, "POPULARITY tabular+BERT")

    pop_tab_df.to_csv(out_dir / "shap_popularity_news_3437_tabular_only.csv", index=False)
    pop_bert_df.to_csv(out_dir / "shap_popularity_news_3437_with_bert.csv", index=False)

    # ---------------- TOPIC ----------------
    onehot = dataset[CHANNEL_COLS]
    has_channel = onehot.sum(axis=1) == 1
    topic_dataset = dataset.loc[has_channel].copy()
    topic_dataset["topic"] = onehot.loc[has_channel].idxmax(axis=1).str.replace("data_channel_is_", "", regex=False)
    print(f"\nTopic dataset (single-channel rows): {len(topic_dataset)} articles")
    print(topic_dataset["topic"].value_counts().to_string())

    le = LabelEncoder()
    y_topic = le.fit_transform(topic_dataset["topic"])
    n_classes = len(le.classes_)
    majority = topic_dataset["topic"].value_counts(normalize=True).max()
    print(f"chance={1/n_classes:.4f}  majority_baseline={majority:.4f}")

    X_bert_topic = load_bert(topic_dataset)
    tab_cols_topic = config.SENTIMENT_FEATURE_COLUMNS + config.LDA_FEATURE_COLUMNS + config.STRUCTURAL_FEATURE_COLUMNS
    X_tab_topic = topic_dataset[tab_cols_topic].to_numpy(dtype=np.float64)

    topic_tab_df, topic_tab_acc = run_one(X_tab_topic, y_topic, tab_cols_topic, TABULAR_GROUPS_TOPIC, "TOPIC tabular-only")

    X_combined_topic = np.hstack([X_tab_topic, X_bert_topic])
    groups_bert_topic = dict(TABULAR_GROUPS_TOPIC)
    groups_bert_topic["bert"] = bert_cols
    topic_bert_df, topic_bert_acc = run_one(X_combined_topic, y_topic, tab_cols_topic + bert_cols, groups_bert_topic, "TOPIC tabular+BERT")

    topic_tab_df.to_csv(out_dir / "shap_topic_news_3437_tabular_only.csv", index=False)
    topic_bert_df.to_csv(out_dir / "shap_topic_news_3437_with_bert.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (full 3,437 scrape)")
    print(f"Popularity: tabular={pop_tab_acc:.4f}  tabular+bert={pop_bert_acc:.4f}  chance=0.2500")
    print(f"Topic:      tabular={topic_tab_acc:.4f}  tabular+bert={topic_bert_acc:.4f}  chance={1/n_classes:.4f}  majority={majority:.4f}")
    print("=" * 70)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
