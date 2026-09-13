"""Prototype-scoring strategies for ``PrototypeIntentStore``.

Each strategy decides two things:

1. **Storage**: which subset / aggregation of a label's sample embeddings
   gets stored as that label's *anchors* at ``add()`` time
   (``select_anchors``).
2. **Scoring**: how a query embedding is turned into one similarity score
   per label at ``scores()`` time (``score_labels``).

Strategies
----------
``mean_centroid``
    One anchor = mean of the per-sample embeddings. ``scores`` = cosine to
    that single centroid. Cheapest storage and inference. The classic
    prototype baseline.

``medoid``
    One anchor = the sample closest to the centroid. Robust to outliers
    when the per-label samples are noisy, no averaging blur.

``max_over_all``
    Every sample (up to ``k``) is kept as its own anchor. Score = max
    cosine over those anchors. Sharpest decision boundary; this is the
    classic *k-nearest-prototype* matcher and matches the pre-strategy
    behaviour of this store (the default).

``top_k_mean``
    All samples are kept; per label, score = mean of the top ``top_k``
    cosines. Combines the sharpness of max-over-all with smoothing.

``farthest_point``
    ``k`` samples per label chosen via maximin (farthest-point sampling)
    so the anchors span the example space. Score = max cosine.

``kmeans_centers``
    ``k`` spherical-k-means centroids per label. Score = max cosine.
    Useful when a label has multi-modal phrasings.

``softmax_weighted``
    All samples kept; per label, score = softmax-weighted average of all
    cosines. Temperature ``tau`` controls sharpness (low = max-like,
    high = mean-like).
"""
from __future__ import annotations

import enum
from typing import Optional

import numpy as np


class PrototypeStrategy(str, enum.Enum):
    """How a label's sample embeddings are aggregated into a match score."""

    MEAN_CENTROID    = "mean_centroid"
    MEDOID           = "medoid"
    MAX_OVER_ALL     = "max_over_all"
    TOP_K_MEAN       = "top_k_mean"
    FARTHEST_POINT   = "farthest_point"
    KMEANS_CENTERS   = "kmeans_centers"
    SOFTMAX_WEIGHTED = "softmax_weighted"


#: Strategies whose ``scores()`` step is a plain per-label max-cosine over
#: the stored anchors. ``select_anchors`` did all the work at ``add()`` time.
_ANCHOR_BASED = {
    PrototypeStrategy.MEAN_CENTROID,
    PrototypeStrategy.MEDOID,
    PrototypeStrategy.MAX_OVER_ALL,
    PrototypeStrategy.FARTHEST_POINT,
    PrototypeStrategy.KMEANS_CENTERS,
}


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(norms > 0, x / norms, x)


def _farthest_point_indices(embs: np.ndarray, k: int) -> list[int]:
    n = len(embs)
    if k >= n:
        return list(range(n))
    centroid = _l2_normalize(embs.mean(0))
    seed = int(np.argmax(embs @ centroid))
    chosen = [seed]
    dists = np.clip(1.0 - embs @ embs[seed], 0.0, None)
    for _ in range(k - 1):
        nxt = int(np.argmax(dists))
        chosen.append(nxt)
        dists = np.minimum(dists, np.clip(1.0 - embs @ embs[nxt], 0.0, None))
    return chosen


def _kmeans_centers(embs: np.ndarray, k: int, seed: int = 0,
                    max_iter: int = 20) -> np.ndarray:
    """Tiny spherical k-means; assumes ``embs`` is L2-normalised."""
    n = len(embs)
    if k >= n:
        return embs.copy()
    rng = np.random.default_rng(seed)
    # k-means++ init
    centres_idx = [int(rng.integers(n))]
    while len(centres_idx) < k:
        dists = np.clip(1.0 - embs @ embs[centres_idx].T, 0.0, None)
        min_d = dists.min(axis=1)
        total = float(min_d.sum())
        if total <= 1e-12:
            remaining = [i for i in range(n) if i not in centres_idx]
            if not remaining:
                break
            centres_idx.append(int(rng.choice(remaining)))
            continue
        centres_idx.append(int(rng.choice(n, p=min_d / total)))
    centres = embs[centres_idx].copy()
    for _ in range(max_iter):
        assignments = np.argmax(embs @ centres.T, axis=1)
        new_centres = np.zeros_like(centres)
        for c in range(len(centres)):
            mask = assignments == c
            new_centres[c] = embs[mask].mean(0) if mask.any() else centres[c]
        new_centres = _l2_normalize(new_centres)
        if np.allclose(centres, new_centres, atol=1e-6):
            break
        centres = new_centres
    return centres


def select_anchors(
    embeddings: np.ndarray,
    strategy: PrototypeStrategy,
    k: Optional[int] = None,
    random_state: int = 42,
) -> np.ndarray:
    """Return the L2-normalised anchors that ``scores()`` will consume.

    Input ``embeddings`` are assumed L2-normalised. Output is also
    L2-normalised. ``k`` caps the anchor count for the subsampling /
    clustering strategies; ``None`` keeps every sample, guaranteeing that
    an exact training sample scores a perfect cosine match. The number of
    returned rows depends on the strategy:

    ==============================  ==============================
    strategy                        anchor count
    ==============================  ==============================
    MEAN_CENTROID                   1
    MEDOID                          1
    MAX_OVER_ALL                    min(k, n)   (random subsample)
    TOP_K_MEAN, SOFTMAX_WEIGHTED    n           (all samples kept)
    FARTHEST_POINT, KMEANS_CENTERS  min(k, n)
    ==============================  ==============================
    """
    n = len(embeddings)
    if n == 0:
        return embeddings
    strategy = PrototypeStrategy(strategy)

    if strategy is PrototypeStrategy.MEAN_CENTROID:
        return _l2_normalize(embeddings.mean(0)).reshape(1, -1)

    if strategy is PrototypeStrategy.MEDOID:
        centroid = _l2_normalize(embeddings.mean(0))
        return embeddings[int(np.argmax(embeddings @ centroid))].reshape(1, -1)

    if k is None:
        # no cap: every remaining strategy keeps all samples
        return embeddings

    if strategy is PrototypeStrategy.MAX_OVER_ALL:
        if n > k:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=k, replace=False)
            return embeddings[idx]
        return embeddings

    if strategy is PrototypeStrategy.FARTHEST_POINT:
        idx = _farthest_point_indices(embeddings, k)
        return embeddings[idx]

    if strategy is PrototypeStrategy.KMEANS_CENTERS:
        return _kmeans_centers(embeddings, k, seed=random_state)

    # TOP_K_MEAN / SOFTMAX_WEIGHTED keep all samples — aggregation
    # happens at score-time, not at storage-time.
    return embeddings


def score_labels(
    query: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    strategy: PrototypeStrategy,
    top_k: int = 3,
    tau: float = 0.1,
) -> dict[str, float]:
    """Return ``{label: score}`` for *query* given the stored anchors.

    Query is assumed L2-normalised. ``embeddings`` and ``labels`` are
    parallel arrays — one row per anchor.
    """
    if len(embeddings) == 0:
        return {}
    strategy = PrototypeStrategy(strategy)
    sims = embeddings @ query  # (n_anchors,)

    if strategy in _ANCHOR_BASED:
        out: dict[str, float] = {}
        for lbl, s in zip(labels, sims):
            lbl = str(lbl)
            if lbl not in out or s > out[lbl]:
                out[lbl] = float(s)
        return out

    # Sample-based strategies: group sims by label, then aggregate.
    by_label: dict[str, list[float]] = {}
    for lbl, s in zip(labels, sims):
        by_label.setdefault(str(lbl), []).append(float(s))

    if strategy is PrototypeStrategy.TOP_K_MEAN:
        return {
            lbl: float(np.sort(np.asarray(v))[-min(top_k, len(v)):].mean())
            for lbl, v in by_label.items()
        }

    if strategy is PrototypeStrategy.SOFTMAX_WEIGHTED:
        out2: dict[str, float] = {}
        for lbl, v in by_label.items():
            arr = np.asarray(v)
            w = np.exp(arr / tau)
            w /= w.sum() + 1e-12
            out2[lbl] = float((w * arr).sum())
        return out2

    raise ValueError(f"unknown strategy: {strategy}")
