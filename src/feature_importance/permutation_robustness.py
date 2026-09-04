"""
Shared helper: group-level permutation importance.

Method-robustness check for the SHAP feature-group rankings already
computed (shap_full_3437.py, shap_reddit.py). Instead of SHAP, use
group permutation importance: shuffle all columns of one group
together across test rows, measure accuracy drop, repeat, average.
Group-level (not per-feature) shuffle keeps cost independent of
BERT's 768 dims -- cheap, no retrain needed beyond the one RF already
being fit for the run.

If SHAP and permutation agree on group ranking -> finding isn't a
SHAP/TreeExplainer artifact.
"""

import numpy as np
from sklearn.metrics import accuracy_score


def group_permutation_importance(clf, X_test, y_test, feature_cols, groups, n_repeats=10, seed=42):
    rng = np.random.default_rng(seed)
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    baseline_acc = accuracy_score(y_test, clf.predict(X_test))

    rows = []
    for group_name, cols in groups.items():
        idxs = [col_idx[c] for c in cols]
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm = rng.permutation(X_perm.shape[0])
            X_perm[:, idxs] = X_perm[perm][:, idxs]
            acc = accuracy_score(y_test, clf.predict(X_perm))
            drops.append(baseline_acc - acc)
        drops = np.asarray(drops)
        rows.append({
            "group": group_name,
            "n_features": len(cols),
            "baseline_acc": baseline_acc,
            "mean_acc_drop": drops.mean(),
            "std_acc_drop": drops.std(),
        })
    return rows
