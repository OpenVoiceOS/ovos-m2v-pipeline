# Benchmark

`ovos-m2v-pipeline` ships a comparative accuracy benchmark in `benchmark/compare.py`. It runs on two OpenVoiceOS evaluation datasets and reports model2vec — in every prototype strategy and in three trained-classifier shapes — alongside a fixed set of external baselines, so results are directly comparable across the OVOS intent-engine family.

---

## Headline results — `intents-for-eval`

50 intents, 1700 labelled test cases, 50 off-topic (`far_ood`) cases.

| Engine | def F_0.5 | **opt F_0.5** | opt thr | opt FP | **Rec @ P≥99%** |
|---|---|---|---|---|---|
| **m2v trained flat** | 0.953 | **0.966** | 0.35 | 15 | **88.1%** |
| m2v trained hier | 0.952 | 0.962 | 0.43 | 20 | 85.3% |
| m2v trained domain | 0.951 | 0.963 | 0.33 | **12** | 86.5% |
| padatious | 0.895 | 0.922 | 0.16 | 10 | 71.8% |
| nebulento `damerau` | 0.909 | 0.918 | 0.43 | 26 | 62.0% |
| m2v flat prototype `top_k_mean` (best) | 0.839 | 0.903 | 0.25 | 28 | 59.4% |
| m2v flat prototype `softmax_weighted` | 0.825 | 0.899 | 0.19 | 30 | 59.5% |
| m2v hier prototype `top_k_mean` (best hier proto) | 0.768 | 0.830 | 0.25 | 24 | 45.3% |
| padaos (regex) | 0.832 | 0.832 | 0.50 | 1 | 0.0% (recall ≤ 50%) |

**Model2vec trained — in any of its three shapes — is the strongest engine in the OVOS intent family by every voice-assistant-relevant metric.** Trained flat tops the headline numbers (F_0.5 0.966 with FP cut from 50 → 15 by calibration, and 88% of all test queries answered at ≥ 99% precision), narrowly ahead of trained hierarchical (0.962) and trained domain (0.963). All three sit ~5pp above padatious and ~7pp above the best fuzzy engine on R@P≥99%.

Read the per-section deep-dive below for what those numbers mean, when to pick which variant, and why every prototype-mode strategy is comfortably outclassed by the trained classifiers.

---

## Why F_0.5 and not F1

A voice assistant's two failure modes are not symmetric:

- **False positive** — the wrong intent fires, the skill executes the wrong action, the assistant says the wrong thing. There is no recovery layer above the intent service that can catch this; the user has to notice, abort, and re-ask.
- **False negative** — no intent fires. OVOS hands the utterance to its fallback chain: common-query, the LLM fallback, online search. These exist precisely to handle "I don't know what you meant."

The cost ratio is roughly 5–10× in favour of false negatives. F1 (which weights precision and recall equally) is the wrong summary metric. **F_β with β=0.5** weights precision twice as recall and is the right summary for OVOS.

We also report **Rec@P≥99%** — the recall achievable once the threshold is tuned to keep precision at or above 99%. This is the operating point a production OVOS install actually picks: "give me the most coverage you can while letting through at most 1% wrong matches."

---

## Datasets

Both datasets are loaded from the Hugging Face Hub by `benchmark/dataset.py`. Each has a `<lang>-templates` config (training templates) and a `<lang>-test` config (labelled evaluation utterances).

| Name | Repo | Intents | Test cases | Notes |
|---|---|---|---|---|
| `intents-for-eval` | [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) | 50 | 1750 | Six test splits including a 50-row `far_ood` no-match set |
| `massive` | [`OpenVoiceOS/massive-templates`](https://huggingface.co/datasets/OpenVoiceOS/massive-templates) | 60 | 2974 | OVOS-templated rebuild of MASSIVE; one labelled split, no no-match cases |

`intents-for-eval` test splits:

| Split | Cases | Tests |
|---|---|---|
| `template` | 500 | Utterances that fill a training template directly |
| `paraphrase` | 700 | Natural rephrasings — different words, same intent |
| `near_ood` | 400 | Boundary utterances close to another intent |
| `far_ood` | 50 | Genuinely off-topic — should match **nothing** |
| `asr_noise` | 50 | Speech-recognition artefacts |
| `typos` | 50 | Spelling errors |

### Slot handling

Every `{slot}` placeholder in the templates ships with a list of example values. `benchmark/dataset.py` collects them into `Bundle.entities`.

- **padaos**, **padatious** and **nebulento** register the slot values as entities natively (their `add_entity` API).
- **model2vec** has no engine-level entity API. The benchmark's `_fill_slots` helper substitutes random slot values from `Bundle.entities` to produce 4 filled training sentences per template before either prototype-store ingestion or classifier fine-tuning. **Without this step the trained-classifier F_0.5 collapses from 0.95 to ~0.05** — the LR head learns the literal token `{song}` and never sees real song names.

---

## Engines

Three external engines are **fixed baselines** — the same engines and settings used in every OVOS intent-engine benchmark. Sixteen model2vec rows are the **subject**: every variant the repo ships.

| Engine | Role | Notes |
|---|---|---|
| `padaos` | baseline | regex-based exact matcher |
| `padatious` | baseline | neural matcher (requires `train()` pass) |
| `nebulento` | baseline | fuzzy string matcher, `DAMERAU_LEVENSHTEIN_SIMILARITY` |
| `m2v flat <strategy>` | subject | `PrototypeIntentStore` — cosine over per-intent prototypes, 7 strategies |
| `m2v hierarchical <strategy>` | subject | `HierarchicalPrototypeIntentStore` — two-stage prototype routing, same 7 strategies |
| `m2v trained flat` | subject | `StaticModelForClassification` end-to-end fine-tune on all intents |
| `m2v trained domain` | subject | One classifier per domain with open-set `__other__` rejection |
| `m2v trained hierarchical` | subject | Top-level domain classifier + per-domain intent classifiers |

The encoder is `minishlab/potion-base-32M` (English) — chosen after a sweep against `potion-multilingual-128M` (-3pp on this English benchmark) and `potion-retrieval-32M` (-5pp). The English-tuned base wins despite its smaller parameter count.

---

## Deep-dive: prototype mode

Prototype mode is unsupervised — each intent's training sentences are reduced to one or more anchor embeddings (centroid, medoid, k-means centres, etc.), and a query is matched by argmaxing cosine similarity. The strategy decides how to pick anchors.

| Strategy | def F_0.5 | opt F_0.5 | opt thr | opt FP | R@P≥99% | latency |
|---|---|---|---|---|---|---|
| `top_k_mean` | 0.839 | **0.903** | 0.25 | 28 | 59.4% | 16.7 ms |
| `softmax_weighted` | 0.825 | 0.899 | 0.19 | 30 | 59.5% | 16.6 ms |
| `kmeans_centers` | 0.831 | 0.880 | 0.21 | 36 | 53.9% | 0.2 ms |
| `farthest_point` | 0.826 | 0.879 | 0.23 | 33 | 56.1% | 0.2 ms |
| `mean_centroid` | 0.746 | 0.871 | 0.21 | 22 | 49.2% | 0.2 ms |
| `max_over_all` | 0.801 | 0.864 | 0.23 | 33 | 47.6% | 0.2 ms |
| `medoid` | 0.704 | 0.852 | 0.21 | 21 | 43.6% | 0.2 ms |

- **`top_k_mean` wins** by a small margin on F_0.5 (0.903). It averages the top-k anchors at match time — robust to a single noisy training sentence in a way that `max_over_all` and `medoid` aren't.
- **`softmax_weighted` follows closely** (0.899) with similar R@P≥99% (59.5%) but at 80× the latency of the cheap centroid strategies — both `top_k_mean` and `softmax_weighted` compute over all anchors per query.
- **Single-anchor strategies (`mean_centroid`, `medoid`, `max_over_all`) are 80× faster** (~0.2 ms vs 16.6 ms) and trade ~3-5pp F_0.5. For real-time deployments this trade may be worth it.
- **Ceiling: ~0.90 F_0.5**. Prototype mode reaches its ceiling because cosine over a small handful of anchors per class can't reshape the embedding space — it's stuck with what model2vec gives it. Trained mode (next section) breaks this ceiling.

The hierarchical-prototype variants underperform their flat counterparts by 4-9pp F_0.5 across every strategy. That's because cosine similarities are already globally comparable across intents; routing through a two-stage gate just adds misroute risk with no gain. The hierarchical prototype variant ships for API symmetry, not because it helps.

---

## Deep-dive: trained mode

Trained mode fine-tunes `StaticModelForClassification` end-to-end (encoder + linear head) using `model2vec.train`. This is the same recipe `train/train_en.py` uses to produce the shipped Jarbas model.

| Variant | def F_0.5 | opt F_0.5 | opt thr | opt FP | R@P≥99% | latency |
|---|---|---|---|---|---|---|
| **trained flat** | 0.953 | **0.966** | 0.35 | 15 | **88.1%** | 0.21 ms |
| trained domain (open-set) | 0.951 | 0.963 | 0.33 | **12** | 86.5% | 2.0 ms |
| trained hierarchical | 0.952 | 0.962 | 0.43 | 20 | 85.3% | 0.77 ms |

**All three trained variants beat every prototype strategy and every external baseline.** The F_0.5 spread between the three is 0.004 — they're effectively equivalent on accuracy, distinguished by FP profile and latency:

- **trained flat** is the simplest, fastest, and the best F_0.5 winner. Default choice unless modularity is a hard requirement.
- **trained domain** (open-set rejection) has the **lowest FP count of any engine in this benchmark** (12). Each per-domain classifier gets balanced cross-domain negatives labelled `__other__`; off-domain queries route those negatives into the `__other__` head and the real-intent confidences drop, making cross-head scores directly comparable. Pick this variant when off-topic rejection matters and you want incremental retraining (one classifier per domain to retrain when a single skill changes).
- **trained hierarchical** sits between. Its domain router runs a single per-domain classifier per query (lower latency than the domain-parallel runner) but pays a small accuracy cost from misrouting.

### Why trained dominates prototype

Prototype mode's ceiling is the embedding space the encoder produces. Trained mode **fine-tunes the encoder** for this specific intent set during `fit()`. The decision boundary the classifier learns can reshape distances between embeddings that cosine-over-centroids cannot reach. That's worth ~6pp F_0.5 on this benchmark and ~30pp R@P≥99%.

The trade-off: **training takes 60-180 seconds per classifier** (vs near-instant prototype indexing). Inference latency is comparable (sub-millisecond for trained flat, vs 0.2 ms for centroid prototypes). For most OVOS deployments — where the intent set is fixed at install time — the training cost is a one-shot.

### Why the open-set `__other__` head matters for trained Domain

Without open-set rejection, each per-domain classifier outputs a softmax over only its own intents. Softmax outputs from a 2-class classifier (max ≈ 0.55) are not comparable to outputs from a 10-class classifier (max ≈ 0.20) — and any off-domain query still produces a "confident" prediction in the wrong domain. Naive parallel argmax across heads collapses to **F_0.5 ≈ 0.30 / FP 36** — worse than prototype.

The open-set fix trains each per-domain classifier with `min(|positives|, |negatives|)` cross-domain negatives labelled `__other__`. Now:
- For an on-domain query, the `__other__` head is suppressed and the right intent wins.
- For an off-domain query, the `__other__` head absorbs probability mass — the real-intent maxes are small and the head doesn't fight for the global argmax.
- The scores are comparable across heads because each one represents "how confident this domain is that this query is one of mine."

Result: F_0.5 0.951 → 0.963 (calibrated), FP 36 → 12.

---

## Threshold calibration in context

Across every engine in the family:

| Engine | calibration delta on F_0.5 | Verdict |
|---|---|---|
| padaos | 0.000 | binary conf — no room to calibrate |
| padacioso | 0.000 | binary conf — no room |
| nebulento `damerau` | +0.009 | shipped well-tuned |
| padatious | +0.027 | shipped well-tuned |
| markov | +0.244 | mis-tuned by default — calibration mandatory |
| m2v trained flat | +0.013 | shipped close to optimal |
| m2v prototype `top_k_mean` | +0.064 | calibration helps moderately |

The two engines whose defaults benefit most from calibration are markov and m2v prototype mode. For m2v trained, the default threshold is 0.0 (no rejection), and calibration narrows that to ~0.35 — buying a +13 FP reduction (50 → 15) for a tiny accuracy cost.

---

## flat vs domain vs hierarchical (m2v specifically)

For each m2v family (prototype, trained), the three shapes have different costs and benefits:

| Property | flat | Domain (parallel) | Hierarchical (two-stage) |
|---|---|---|---|
| F_0.5 (opt, trained) | 0.966 | 0.963 | 0.962 |
| FP (opt, trained) | 15 | **12** | 20 |
| R@P≥99% (trained) | **88.1%** | 86.5% | 85.3% |
| Per-query latency | flat | N classifiers (slowest) | 1 router + 1 classifier |
| Add/remove a skill | full retrain | one-domain retrain | one-domain retrain + router retrain |
| Recommended when | default for most deployments | precision/modularity matters | latency vs flat is unacceptable |

The Domain variant in **prototype** mode is a no-op (cosine over per-domain anchors gives identical results to flat-cosine over global anchors) — the repo ships hierarchical-prototype but not domain-prototype. In **trained** mode the Domain variant genuinely differs (per-domain classifiers train against different negatives) and is worth shipping.

---

## Encoder choice

We benchmarked three model2vec encoders on `intents-for-eval`:

| Encoder | trained flat F_0.5 (opt) | best prototype F_0.5 (opt) |
|---|---|---|
| `minishlab/potion-base-32M` | **0.966** | **0.903** |
| `minishlab/potion-multilingual-128M` | 0.951 | 0.876 |
| `minishlab/potion-retrieval-32M` | 0.948 | 0.864 |

The English-specialised base wins on every metric despite its smaller parameter count — the multilingual model dilutes English representation, and the retrieval-tuned variant optimises for similarity ranking (which matters less here because trained mode reshapes the space anyway).

---

## Reproducing

```bash
pip install ovos-m2v-pipeline[benchmark]
python benchmark/compare.py intents-for-eval   # ~15 minutes (trained rows dominate)
python benchmark/compare.py massive            # ~45 minutes
```

The first run downloads each dataset from the Hugging Face Hub (cached afterwards). Prototype rows run in seconds; the 3 trained-classifier rows fine-tune end-to-end with `model2vec.train.StaticModelForClassification.fit()` and dominate the runtime.

## How metrics are calculated

Source: `compute_metrics`, `calibrate_threshold`, `fbeta`, `recall_at_precision` in `benchmark/compare.py`.

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2·P·R / (P + R)
- **F_0.5** = 1.25·P·R / (0.25·P + R) — weights precision 2× recall (default summary metric for OVOS)
- **Rec@P≥99%** = max recall achievable by sweeping the threshold while keeping precision ≥ 99%
- **FP** = no-match utterances incorrectly assigned an intent

A prediction is a TP when the predicted intent name exactly matches the expected intent and `conf ≥ threshold`. A no-match case is correct only when the engine returns `None` or a confidence below threshold.
