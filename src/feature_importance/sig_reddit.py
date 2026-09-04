"""
Significance check -- Reddit domain, full 15,998 posts.
Same RF (max_depth=18 capped), same split as shap_reddit.py /
robustness_reddit.py. Paired bootstrap CI on accuracy delta
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
import config  # noqa: E402
import data_loader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from significance_check import paired_bootstrap_ci  # noqa: E402

RANDOM_SEED = config.RANDOM_SEED
BERT_DIM = 768


def load_bert(dataset):
    meta = pd.read_csv(config.BERT_EMBEDDING_METADATA_PATH)
    emb = np.load(config.BERT_EMBEDDINGS_PATH)
    id_to_row = {row_id: i for i, row_id in enumerate(meta["id"])}
    idx = dataset["id"].map(id_to_row).to_numpy()
    return emb[idx]


def fit_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=18, random_state=RANDOM_SEED, n_jobs=-1)
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
    print("SIGNIFICANCE CHECK -- REDDIT -- POPULARITY + TOPIC")
    print("=" * 70)

    dataset = data_loader.load_feature_prediction_dataset()
    pop_cols = list(config.META_FEATURE_COLUMNS)
    X_tab = dataset[pop_cols].to_numpy(dtype=np.float64)
    y_pop = dataset["popularity_class"].to_numpy(dtype=np.int64)
    X_bert = load_bert(dataset)
    X_combined = np.hstack([X_tab, X_bert])

    pop_result = check("POPULARITY (reddit, 15998)", X_tab, X_combined, y_pop)

    le = LabelEncoder()
    y_topic = le.fit_transform(dataset["subreddit"])
    topic_cols = [c for c in pop_cols if c != "subreddit_code"]
    X_tab_topic = dataset[topic_cols].to_numpy(dtype=np.float64)
    X_combined_topic = np.hstack([X_tab_topic, X_bert])

    topic_result = check("TOPIC (reddit, subreddit)", X_tab_topic, X_combined_topic, y_topic)

    out_dir = config.OUTPUTS_DIR / "feature_importance"
    pd.DataFrame([{"task": "popularity", **pop_result}, {"task": "topic", **topic_result}]).to_csv(
        out_dir / "significance_reddit_full.csv", index=False
    )
    print(f"\nSaved to {out_dir / 'significance_reddit_full.csv'}")


if __name__ == "__main__":
    main()
