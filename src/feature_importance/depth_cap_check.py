"""Capped (max_depth=18) vs uncapped RF: SHAP group ranking on the same
1,000-post Reddit subset shap_reddit.py uses for its sanity run.
Reuses shap_reddit.run_one; only max_depth differs. Writes nothing to outputs/."""
import contextlib, io, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "feature_importance"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config, data_loader
import shap_reddit as sr

for p in (config.BERT_EMBEDDINGS_PATH, config.BERT_EMBEDDING_METADATA_PATH):
    assert Path(p).exists(), p

ds = data_loader.load_feature_prediction_dataset()
ds, _ = train_test_split(ds, train_size=1000, stratify=ds["popularity_class"], random_state=config.RANDOM_SEED)
X_bert = sr.load_bert(ds)
bert_cols = [f"bert_{i}" for i in range(sr.BERT_DIM)]

pop_cols = list(config.META_FEATURE_COLUMNS)
topic_cols = [c for g in sr.FEATURE_GROUPS_TOPIC.values() for c in g]
y_pop = ds["popularity_class"].to_numpy(dtype=np.int64)
y_topic = LabelEncoder().fit_transform(ds["subreddit"])
cells = {
    "popularity tabular_only": (ds[pop_cols].to_numpy(float), y_pop, pop_cols, sr.FEATURE_GROUPS_POPULARITY),
    "popularity with_bert": (np.hstack([ds[pop_cols].to_numpy(float), X_bert]), y_pop, pop_cols + bert_cols,
                             {**sr.FEATURE_GROUPS_POPULARITY, "bert": bert_cols}),
    "topic tabular_only": (ds[topic_cols].to_numpy(float), y_topic, topic_cols, sr.FEATURE_GROUPS_TOPIC),
    "topic with_bert": (np.hstack([ds[topic_cols].to_numpy(float), X_bert]), y_topic, topic_cols + bert_cols,
                        {**sr.FEATURE_GROUPS_TOPIC, "bert": bert_cols}),
}

def run(depth, X, y, cols, groups):
    sr.RandomForestClassifier = lambda **kw: RandomForestClassifier(**{**kw, "max_depth": depth})
    with contextlib.redirect_stdout(io.StringIO()):
        df, acc = sr.run_one(X, y, cols, groups, "")
    return df, acc

all_match = True
for name, args in cells.items():
    rank = {}
    for depth in (18, None):
        df, acc = run(depth, *args)
        order = list(df.sort_values("pct_of_avg_per_feature", ascending=False)["group"])
        raw = list(df.sort_values("pct_of_total", ascending=False)["group"])
        rank[depth] = (order, raw)
        print(f"{name:26s} depth={str(depth):4s} acc={acc:.3f} per-feat={order} raw={raw}")
    same = rank[18] == rank[None]
    all_match &= same
    print(f"{'':26s} -> identical rankings: {same}")
print("ALL CELLS IDENTICAL:", all_match)
