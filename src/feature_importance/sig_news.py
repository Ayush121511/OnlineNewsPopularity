"""
Significance check -- news domain, full 3,437 scrape.
Same RF, same split (random_seed) as shap_full_3437.py /
robustness_news.py. Paired bootstrap CI on accuracy delta
(tabular+bert - tabular-only), both tasks.
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
from significance_check import paired_bootstrap_ci  # noqa: E402

RANDOM_SEED = config.RANDOM_SEED
BERT_DIM = 768
CHANNEL_COLS = config.CHANNEL_FEATURE_COLUMNS


def load_bert(dataset):
    meta = pd.read_csv(config.OUTPUTS_DIR / "feature_importance" / "news_full_bert_embedding_metadata.csv")
    emb = np.load(config.OUTPUTS_DIR / "feature_importance" / "news_full_bert_embeddings.npy")
    id_to_row = {row_id: i for i, row_id in enumerate(meta["id"])}
    idx = dataset["id"].map(id_to_row).to_numpy()
    return emb[idx]


def fit_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    return y_test, clf.predict(X_test)


def check(label, X_tab, X_combined, y):
    y_test_a, preds_a = fit_predict(X_tab, y)
    y_test_b, preds_b = fit_predict(X_combined, y)
    assert np.array_equal(y_test_a, y_test_b), "test split mismatch -- pairing invalid"
    result = paired_bootstrap_ci(y_test_a, preds_a, preds_b, n_boot=2000, seed=RANDOM_SEED)
    print(f"\n--- {label} ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def main():
    print("=" * 70)
    print("SIGNIFICANCE CHECK -- NEWS 3,437 -- POPULARITY + TOPIC")
    print("=" * 70)

    dataset = data_loader.load_feature_prediction_dataset()
    tab_cols = config.MODEL_FEATURE_COLUMNS
    X_tab = dataset[tab_cols].to_numpy(dtype=np.float64)
    y_pop = dataset["popularity_class"].to_numpy(dtype=np.int64)
    X_bert = load_bert(dataset)
    X_combined = np.hstack([X_tab, X_bert])

    pop_result = check("POPULARITY (news, 3437)", X_tab, X_combined, y_pop)

    onehot = dataset[CHANNEL_COLS]
    has_channel = onehot.sum(axis=1) == 1
    topic_dataset = dataset.loc[has_channel].copy()
    topic_dataset["topic"] = onehot.loc[has_channel].idxmax(axis=1).str.replace("data_channel_is_", "", regex=False)
    le = LabelEncoder()
    y_topic = le.fit_transform(topic_dataset["topic"])
    tab_cols_topic = config.SENTIMENT_FEATURE_COLUMNS + config.LDA_FEATURE_COLUMNS + config.STRUCTURAL_FEATURE_COLUMNS
    X_tab_topic = topic_dataset[tab_cols_topic].to_numpy(dtype=np.float64)
    X_bert_topic = load_bert(topic_dataset)
    X_combined_topic = np.hstack([X_tab_topic, X_bert_topic])

    topic_result = check("TOPIC (news, single-channel)", X_tab_topic, X_combined_topic, y_topic)

    out_dir = config.OUTPUTS_DIR / "feature_importance"
    pd.DataFrame([{"task": "popularity", **pop_result}, {"task": "topic", **topic_result}]).to_csv(
        out_dir / "significance_news_3437.csv", index=False
    )
    print(f"\nSaved to {out_dir / 'significance_news_3437.csv'}")


if __name__ == "__main__":
    main()
