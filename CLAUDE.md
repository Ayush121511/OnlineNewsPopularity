# Project: Cross-Domain BERT Feature-Importance Study

Test whether frozen BERT embeddings add predictive signal beyond
structural/meta features, across two tasks (popularity, topic
classification) and two domains (news articles — Mashable/UCI, Reddit
posts — Top 2.5M).

Existing pipeline: data_loader, bert_embeddings (DistilBERT, masked
mean pooling), tabular classifiers, stacking ensemble (tabular+BERT
via OOF probs).

Prior finding: BERT near-chance on popularity (both domains), 12x
chance on topic (validated control).

Next phase: SHAP/permutation importance per feature group (temporal,
source, structure, meta) x task x dataset, with/without BERT block,
to map where text signal does/doesn't add value.

Method follows AutoGluon multimodal-AutoML precedent (Shi et al.
2021) — not claiming novel architecture, claiming novel empirical
finding.

## Conventions
- Keep `src/` layout as-is: data_loader.py, config.py,
  bert_embeddings.py, classifiers.py, preprocessing.py per branch
  (traditional_ml/, bert_popularity/, stacking_model/,
  combined_model/, feature_importance/).
- Fixed seed/split protocol (config.RANDOM_SEED, 80/10/10) must stay
  consistent across all new scripts.
- Confirm data files + precomputed outputs/bert_embeddings.npy exist
  via config.py paths before running — never hardcode paths.
- Run scoped tasks (one feature group / one dataset at a time), not
  one large script.

## Communication style
Use caveman mode (full level) for all responses in this project:
terse, technical substance kept, no filler/hedging/pleasantries, drop
articles, short synonyms, fragments OK. No tool-call narration. Numbers/
units/negations exact, code and error strings verbatim. See caveman
skill for full rules.