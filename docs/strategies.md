# Prototype Strategies

`PrototypeStrategy` (`ovos_m2v_pipeline/strategies.py:51`) is a `str`-enum that controls how a label's training-time embeddings are compressed into stored anchors and how those anchors are turned into a match score at inference time.

The strategy is selected via the `prototype_strategy` config key (default `"max_over_all"`). See [Configuration](configuration.md#prototype-strategies-prototype-mode-only).

## How strategies work

Every strategy defines two phases:

- **Storage** (`select_anchors` in `ovos_m2v_pipeline/strategies.py:131`) runs at `PrototypeIntentStore.add()` time. It decides which subset or aggregation of the sample embeddings is persisted as *anchors* for the label.
- **Scoring** (`score_labels` in `ovos_m2v_pipeline/strategies.py:183`) runs at inference time. It takes the query embedding and the stored anchors and returns one `float` score per label.

Both functions accept L2-normalised input and return L2-normalised output / scores.

## Strategy reference

| Value | Anchors stored | Score per label |
|-------|---------------|-----------------|
| `max_over_all` | Every sample (random subsample only when `prototype_k` is set and `n > k`) | Max cosine over anchors |
| `mean_centroid` | 1, the mean of all samples, re-normalized | Cosine to centroid |
| `medoid` | 1, the sample closest to centroid | Cosine to medoid |
| `top_k_mean` | All samples | Mean of top-`prototype_top_k` cosines |
| `farthest_point` | Every sample, or `prototype_k` samples via maximin (farthest-point) sampling when set | Max cosine |
| `kmeans_centers` | Every sample, or `prototype_k` spherical k-means centroids when set | Max cosine |
| `softmax_weighted` | All samples | Softmax-weighted average, temperature = `prototype_tau` |

### Notes

- `max_over_all` is the default. It is byte-compatible with the store's behavior before strategies were introduced, so existing `.npz` files and tuned confidence thresholds remain valid.
- `mean_centroid` and `medoid` always store exactly one anchor per label, making them cheapest at inference time.
- `top_k_mean` and `softmax_weighted` keep every sample, so memory cost scales with corpus size. Both do their aggregation at score time.
- `farthest_point` and `kmeans_centers` cluster examples to span the label's semantic space. This helps when a skill's intent file contains phrasings from distinct semantic clusters.
- For `softmax_weighted`, lower `prototype_tau` values make the score approximate the maximum cosine. Higher values approach the mean cosine.

## Direct API

The two helpers can also be called directly, independently of `PrototypeIntentStore`:

```python
import numpy as np
from ovos_m2v_pipeline.strategies import PrototypeStrategy, select_anchors, score_labels

# select_anchors: storage phase
anchors = select_anchors(embeddings, PrototypeStrategy.KMEANS_CENTERS, k=4)

# score_labels: scoring phase
scores = score_labels(
    query,
    all_anchors,
    all_labels,
    PrototypeStrategy.SOFTMAX_WEIGHTED,
    top_k=3,
    tau=0.05,
)
```

Both functions accept and return `np.ndarray` with `float32` values. `embeddings` / `query` are assumed to be L2-normalised on input.

---
[← Pipeline Internals](pipeline.md) · [Home](README.md) · [Models →](models.md)
