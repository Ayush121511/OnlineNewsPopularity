# When Does BERT Help? A Cross-Domain Study of the Marginal Value of Text Embeddings Beyond Structured Features

Code and data pipeline for the paper accepted at ICAIA 2026 (IEEE). Tests whether
frozen DistilBERT embeddings add predictive signal beyond structural/meta
features, across two prediction tasks (popularity, topic classification) and
two domains (news articles, Reddit posts), using SHAP feature-group
importance, permutation-importance as an independent cross-check, and a
paired-bootstrap significance test on the tabular-only vs. tabular+BERT
accuracy delta.

The author's accepted version of the paper is in
`paper/Srivastava_ICAIA2026_accepted.pdf`.

## Result summary

| Domain | Task | Tabular-only | +BERT | Δ [95% CI] |
|---|---|---|---|---|
| News | Popularity | 36.3% | 32.1% | −4.2 (n.s.) |
| News | Topic | 79.9% | 77.8% | −2.1 (n.s.) |
| Reddit | Popularity | 33.6% | 30.8% | **−2.9** (p = 0.008) |
| Reddit | Topic | 64.7% | 82.5% | **+17.8** (p < 0.001) |

BERT helps significantly on exactly one of four cells — Reddit subreddit
classification, where the label is intrinsically lexical and no engineered
text-derived feature already proxies for it. It hurts or has no significant
effect everywhere else, including a case with 4.6x more training data.

## Repository layout

```
data/                       Raw datasets (UCI Online News Popularity,
                             Reddit popularity/topic sample, scraped
                             article text + retrieval metadata)
src/
  config.py                 Reddit-domain configuration and paths
  config_news.py            News-domain configuration and paths
  data_loader.py            Reddit data loading / feature grouping
  data_loader_news.py       News data loading / feature grouping
  bert_embeddings.py        Frozen DistilBERT embedding extraction
                             (masked mean pooling)
  feature_importance/       SHAP, permutation importance, and
                             paired-bootstrap significance scripts —
                             the core pipeline behind the paper's results
  scrapper.py                Article re-scraping for the news domain
                             (live fetch + Wayback Machine fallback)
outputs/
  feature_importance/       SHAP / permutation / significance CSVs for
                             all four (domain x task) cells, tabular-only
                             and tabular+BERT
  bert_embeddings.npy        Precomputed embeddings
paper/
  Srivastava_ICAIA2026_accepted.pdf   Author's accepted version (IEEE
                             copyright notice on page 1)
  generate_figures.py        Regenerates all paper figures from
                             outputs/feature_importance/
  figures/                    Generated figures (PDF)
```

## Reproducing the results

```bash
conda env create -f environment.yml
conda activate online_news_popularity

# Embeddings (frozen DistilBERT, masked mean pooling)
python src/bert_embeddings.py

# Per-cell SHAP / permutation-importance / significance (repeat per
# domain x task; see src/feature_importance/ for the four scripts)
python src/feature_importance/shap_full_3437.py        # news
python src/feature_importance/shap_reddit.py            # reddit
python src/feature_importance/robustness_news.py        # permutation, news
python src/feature_importance/robustness_reddit.py      # permutation, reddit
python src/feature_importance/sig_news.py                # bootstrap CI, news
python src/feature_importance/sig_reddit.py               # bootstrap CI, reddit
python src/feature_importance/depth_cap_check.py          # max_depth cap vs uncapped, reddit

# Figures
python paper/generate_figures.py
```

All splits use a fixed seed (42), stratified 80/20, shared across every
script. See Section IV ("Method") of the paper for the full protocol.

## Data

- **News**: [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity)
  (Fernandes et al., 2015), plus re-scraped article text for a 3,437-article
  subset (`data/scraped_articles.csv`, `data/retrieval_metadata.csv`).
- **Reddit**: a balanced 16-subreddit sample of ~16,000 posts, sampled
  down from a raw pull of ~115,000 submissions
  (`data/reddit_popularity_dataset.csv`) with standard submission-level
  fields (title, selftext, score, comment count, timing, NSFW flag,
  thumbnail/domain indicators). The original acquisition source is not
  preserved in this repository; the collection script that built this
  file is no longer available.

## Citation

Accepted at the 2026 4th International Conference on Artificial
Intelligence and Applications (ICAIA), IEEE. The full citation and DOI
will be added here once the paper is published in IEEE Xplore.

© 2026 IEEE. Personal use of the accepted version is permitted. Permission
from IEEE must be obtained for all other uses.
