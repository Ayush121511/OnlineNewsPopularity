"""
Shared helper: paired bootstrap CI on accuracy delta between two
classifiers (tabular-only vs tabular+BERT) evaluated on the SAME
test set (identical split, identical row order -> predictions are
paired). Resample test-row indices with replacement, recompute both
accuracies each resample, collect delta distribution.

95% CI excluding 0 -> delta significant at alpha=0.05.
p-value = 2 * min(frac(delta<=0), frac(delta>=0)) (two-sided, from
the bootstrap distribution itself).
"""

import numpy as np


def paired_bootstrap_ci(y_test, preds_a, preds_b, n_boot=2000, seed=42):
    y_test = np.asarray(y_test)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = len(y_test)
    rng = np.random.default_rng(seed)

    correct_a = (preds_a == y_test).astype(np.float64)
    correct_b = (preds_b == y_test).astype(np.float64)

    acc_a = correct_a.mean()
    acc_b = correct_b.mean()
    observed_delta = acc_b - acc_a

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = correct_b[idx].mean() - correct_a[idx].mean()

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    frac_le0 = (deltas <= 0).mean()
    frac_ge0 = (deltas >= 0).mean()
    p_value = 2 * min(frac_le0, frac_ge0)
    p_value = min(p_value, 1.0)

    return {
        "acc_tabular": acc_a,
        "acc_tabular_bert": acc_b,
        "observed_delta": observed_delta,
        "ci_lo_95": lo,
        "ci_hi_95": hi,
        "significant_at_05": (lo > 0) or (hi < 0),
        "p_value_bootstrap": p_value,
    }
