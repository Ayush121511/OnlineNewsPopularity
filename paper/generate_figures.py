"""
Generate all figures for the paper from the actual result CSVs in
outputs/feature_importance/. Run once; figures are saved as PDF
(vector) into paper/figures/ for inclusion in the IEEEtran LaTeX doc.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "feature_importance"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
})

TAB_COLOR = "#4C72B0"
BERT_COLOR = "#DD8452"
SIG_COLOR = "#55A868"
NS_COLOR = "#999999"

# ============================================================
# Figure 1: headline accuracy comparison, all 4 task x domain cells
# ============================================================

sig_news = pd.read_csv(OUT / "significance_news_3437.csv")
sig_reddit = pd.read_csv(OUT / "significance_reddit_full.csv")

cells = [
    ("News\nPopularity", sig_news.iloc[0], 0.25, "4-way"),
    ("News\nTopic", sig_news.iloc[1], 1 / 6, "6-way"),
    ("Reddit\nPopularity", sig_reddit.iloc[0], 0.25, "4-way"),
    ("Reddit\nTopic", sig_reddit.iloc[1], 1 / 16, "16-way"),
]

fig, ax = plt.subplots(figsize=(7.1, 2.7))
x = np.arange(len(cells))
width = 0.32

tab_accs = [c[1]["acc_tabular"] for c in cells]
bert_accs = [c[1]["acc_tabular_bert"] for c in cells]
chances = [c[2] for c in cells]
sig_flags = [c[1]["significant_at_05"] for c in cells]

b1 = ax.bar(x - width / 2, tab_accs, width, label="Tabular only", color=TAB_COLOR, zorder=3)
b2 = ax.bar(x + width / 2, bert_accs, width, label="Tabular + BERT", color=BERT_COLOR, zorder=3)

for xi, ch in zip(x, chances):
    ax.plot([xi - width, xi + width], [ch, ch], color="black", linestyle="--", linewidth=0.9, zorder=4)

for xi, ta, ba, sig in zip(x, tab_accs, bert_accs, sig_flags):
    top = max(ta, ba) + 0.025
    label = "BERT effect:\nsignificant" if sig else "BERT effect:\nnot significant"
    color = SIG_COLOR if sig else NS_COLOR
    ax.text(xi, top, label, ha="center", va="bottom", fontsize=6.6, color=color, fontweight="bold" if sig else "normal")

ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in cells])
ax.set_ylabel("Test accuracy")
ax.set_ylim(0, 1.13)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(loc="upper left", ncol=1, framealpha=0.9)
ax.set_title("Tabular-only vs. tabular+BERT accuracy across the $2\\times2$ study grid\n"
              "(dashed line = chance level; significance = paired bootstrap test, $\\alpha=0.05$, $n{=}2000$ resamples)",
              fontsize=8.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_accuracy_summary.pdf")
plt.close(fig)

# ============================================================
# Figure 2: bootstrap CI forest plot of accuracy delta
# ============================================================

rows = [
    ("News – Popularity", sig_news.iloc[0]),
    ("News – Topic", sig_news.iloc[1]),
    ("Reddit – Popularity", sig_reddit.iloc[0]),
    ("Reddit – Topic", sig_reddit.iloc[1]),
]

fig, ax = plt.subplots(figsize=(5.2, 2.6))
ypos = np.arange(len(rows))[::-1]
for y, (name, r) in zip(ypos, rows):
    lo, hi, mid = r["ci_lo_95"], r["ci_hi_95"], r["observed_delta"]
    color = SIG_COLOR if r["significant_at_05"] else NS_COLOR
    ax.plot([lo, hi], [y, y], color=color, linewidth=2.2, zorder=3)
    ax.plot(mid, y, "o", color=color, markersize=5, zorder=4)

ax.axvline(0, color="black", linewidth=0.9, linestyle="--", zorder=1)
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel(r"Accuracy delta (tabular+BERT $-$ tabular), 95% bootstrap CI")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("BERT effect on accuracy: paired bootstrap 95% CI ($n{=}2000$ resamples)", fontsize=8.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_ci_forest.pdf")
plt.close(fig)

# ============================================================
# Figures 3-4: per-feature-normalized SHAP group importance,
# news domain and reddit domain (2 tasks each, tabular vs +bert)
# ============================================================

GROUP_COLORS = {
    "sentiment": "#8172B2", "lda": "#C44E52", "channel": "#CCB974",
    "structural": "#4C72B0", "structure": "#4C72B0",
    "temporal": "#8172B2", "source": "#CCB974", "meta": "#64B5CD",
    "bert": "#DD8452",
}


def plot_shap_domain(domain_label, file_prefix, panels, outfile):
    fig, axes = plt.subplots(1, len(panels), figsize=(7.1, 2.9), sharey=False)
    for ax, (title, tab_csv, bert_csv) in zip(axes, panels):
        tab = pd.read_csv(OUT / tab_csv).sort_values("avg_mean_abs_shap_per_feature", ascending=True)
        bert = pd.read_csv(OUT / bert_csv).sort_values("avg_mean_abs_shap_per_feature", ascending=True)

        groups_order = list(bert["group"])
        y = np.arange(len(groups_order))

        bert_vals = bert.set_index("group").loc[groups_order, "pct_of_avg_per_feature"]
        colors = [GROUP_COLORS.get(g, "#333333") for g in groups_order]
        ax.barh(y, bert_vals, color=colors, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(groups_order, fontsize=7.6)
        ax.set_xlabel("% of per-feature SHAP\n(tabular+BERT model)", fontsize=7.6)
        ax.set_title(title, fontsize=8.5)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    fig.suptitle(f"{domain_label}: per-feature-normalized SHAP group importance (with BERT)", fontsize=8.8, y=1.03)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


plot_shap_domain(
    "News (n=3{,}437 / 2{,}885)", "news",
    [
        ("Popularity", "shap_popularity_news_3437_tabular_only.csv", "shap_popularity_news_3437_with_bert.csv"),
        ("Topic (channel)", "shap_topic_news_3437_tabular_only.csv", "shap_topic_news_3437_with_bert.csv"),
    ],
    FIG_DIR / "fig_shap_news.pdf",
)

plot_shap_domain(
    "Reddit (n=15{,}998)", "reddit",
    [
        ("Popularity", "shap_popularity_reddit_full_tabular_only.csv", "shap_popularity_reddit_full_with_bert.csv"),
        ("Topic (subreddit)", "shap_topic_reddit_full_tabular_only.csv", "shap_topic_reddit_full_with_bert.csv"),
    ],
    FIG_DIR / "fig_shap_reddit.pdf",
)

# ============================================================
# Figure 5: SHAP vs. permutation-importance method agreement
# (BERT group's rank position vs its normalized importance,
# across the 4 tabular+BERT runs, both methods)
# ============================================================

configs = [
    ("News\nPopularity", "shap_popularity_news_3437_with_bert.csv", "perm_popularity_news_3437_with_bert.csv"),
    ("News\nTopic", "shap_topic_news_3437_with_bert.csv", "perm_topic_news_3437_with_bert.csv"),
    ("Reddit\nPopularity", "shap_popularity_reddit_full_with_bert.csv", "perm_popularity_reddit_full_with_bert.csv"),
    ("Reddit\nTopic", "shap_topic_reddit_full_with_bert.csv", "perm_topic_reddit_full_with_bert.csv"),
]

shap_bert_pct, perm_bert_pct = [], []
for _, shap_f, perm_f in configs:
    shap_df = pd.read_csv(OUT / shap_f)
    perm_df = pd.read_csv(OUT / perm_f)
    shap_bert_pct.append(shap_df.set_index("group").loc["bert", "pct_of_avg_per_feature"])
    perm_row = perm_df.set_index("group").loc["bert"]
    perm_bert_pct.append(max(perm_row["pct_of_total_drop"], 0.0))

fig, ax = plt.subplots(figsize=(3.4, 2.9))
xw = np.arange(len(configs))
width = 0.32
bars_shap = ax.bar(xw - width / 2, shap_bert_pct, width, label="SHAP (per-feature %)", color="#4C72B0", zorder=3)
bars_perm = ax.bar(xw + width / 2, perm_bert_pct, width, label="Permutation (% of drop)", color="#DD8452", zorder=3)

# Value labels on every bar -- some SHAP bars (e.g. News Topic, 1.2%) are
# too short to read against a 68% neighbor on a linear axis, so annotate
# the exact number directly rather than relying on bar height alone.
for bars in (bars_shap, bars_perm):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.3)

ax.set_xticks(xw)
ax.set_xticklabels([c[0] for c in configs], fontsize=7.2)
ax.set_ylabel("BERT group importance share")
ax.set_ylim(0, 78)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
ax.legend(loc="upper left", framealpha=0.9, fontsize=6.8)
ax.set_title("BERT's importance share: SHAP vs.\npermutation importance agree", fontsize=8.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_method_agreement.pdf")
plt.close(fig)

# ============================================================
# Figure 6: dataset class balance (small, single column)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0))
sys.path.insert(0, str(ROOT / "src"))
import config_news, data_loader_news  # noqa: E402
import config as config_reddit, data_loader as data_loader_reddit  # noqa: E402

news_ds = data_loader_news.load_feature_prediction_dataset()
reddit_ds = data_loader_reddit.load_feature_prediction_dataset()

news_counts = news_ds["popularity_class"].value_counts().sort_index()
reddit_counts = reddit_ds["popularity_class"].value_counts().sort_index()

axes[0].bar(news_counts.index.astype(str), news_counts.values, color=TAB_COLOR)
axes[0].set_title("News (n=3,437)", fontsize=8)
axes[0].set_xlabel("Popularity class", fontsize=7.5)
axes[0].set_ylabel("Count", fontsize=7.5)

axes[1].bar(reddit_counts.index.astype(str), reddit_counts.values, color=BERT_COLOR)
axes[1].set_title("Reddit (n=15,998)", fontsize=8)
axes[1].set_xlabel("Popularity class", fontsize=7.5)

fig.suptitle("Popularity-class balance (both domains, quartile-derived)", fontsize=8, y=1.05)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_class_balance.pdf", bbox_inches="tight")
plt.close(fig)

print("All figures written to", FIG_DIR)
for f in sorted(FIG_DIR.glob("*.pdf")):
    print(" -", f.name)
