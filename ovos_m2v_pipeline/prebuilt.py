"""Export and import prebuilt prototype-mode centroids.

Prototype mode (``ovos-m2v-prototype-pipeline``) builds its label set at
runtime by encoding every registered skill's example utterances through the
embedding model. On a constrained device (a Raspberry Pi, say) that encode
pass is the expensive part of boot -- the artifact this module reads and
writes lets a developer run it once, on a desktop, and ship the *result*
(L2-normalised centroid vectors, not raw text) to every device that needs the
same labels.

The artifact is a directory::

    <out_dir>/
        prototypes.npz   # embeddings + internal (label, lang) keys --
                          # exactly PrototypeIntentStore's own save() format
        manifest.json     # provenance + the per-label cache key each entry
                          # was built under

Loading it never bypasses the invalidation rule the live on-disk cache
already uses (``ovos_m2v_pipeline.cache.compute_cache_key``): a prebuilt
entry is only used in place of encoding when a live registration's cache key
matches the manifest's recorded key for that label -- the same condition
that would make the on-disk ``PrototypeCache`` a hit. A prebuilt directory is
therefore invalidated by exactly the same things (a changed template, a
different model, a bumped ``model2vec``) as the live cache, never by a
separate rule.

The embedding model itself is not part of the artifact and is still needed
at query time: matching an incoming utterance means embedding *it* too, live.
Only the one-time, per-label registration encode is skipped for labels the
artifact already covers.

Loading the artifact never installs its vectors as the live match set on its
own: a label only becomes matchable once a loaded skill actually registers
it, at which point matching cache-key vectors are copied in instead of
encoding. A label the artifact carries with no corresponding live
registration -- a skill that is not installed, or one that renamed or
dropped the intent since the artifact was built -- stays absent from the
live store and can never match, the same as without a prebuilt artifact.
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ovos_utils.log import LOG

#: bumped only if the artifact's on-disk shape changes incompatibly
MANIFEST_FORMAT_VERSION = 1


def export_store(
    store,
    out_dir: Union[str, Path],
    *,
    model_id: str,
    model2vec_version: str,
    cache_keys: Optional[Dict[str, str]] = None,
    plugin_version: str = "",
) -> Path:
    """Write *store* out as a prebuilt artifact directory. Returns *out_dir*.

    *cache_keys* maps each of the store's bare labels to the cache key its
    live registration would compute (``compute_cache_key``); entries with no
    known key are still exported (usable by ``PrototypeIntentStore.load``)
    but will never satisfy the fast-path skip-the-encoder check on import,
    since there is nothing to compare a live registration's key against.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = store.embeddings  # triggers _consolidate()
    internal_labels = store._labels.astype(str) if len(embeddings) else np.array([], dtype=str)
    np.savez(out_dir / "prototypes.npz", embeddings=embeddings, labels=internal_labels)

    labels = sorted(str(l) for l in store.unique_labels)
    dim = int(embeddings.shape[1]) if embeddings.ndim == 2 and len(embeddings) else 0
    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "model_id": model_id,
        "model2vec_version": model2vec_version,
        "embedding_dim": dim,
        "prototype_strategy": store.strategy.value,
        "prototype_top_k": store.top_k,
        "prototype_tau": store.tau,
        "plugin_version": plugin_version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labels": labels,
        "languages": {
            label: sorted(langs)
            for label, langs in sorted(store._langs_by_label.items())
        },
        "cache_keys": dict(sorted((cache_keys or {}).items())),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    LOG.info(
        f"exported {len(labels)} prebuilt prototype label(s), "
        f"{len(embeddings)} vector(s) -> {out_dir}"
    )
    return out_dir


def resolve_prebuilt_path(path_or_repo_id: str) -> Path:
    """Resolve *path_or_repo_id* to a local artifact directory.

    A local, already-existing directory is returned unchanged. Anything
    else is treated as a Hugging Face Hub dataset (falling back to model)
    repo id and fetched via ``huggingface_hub.snapshot_download`` into the
    shared HF cache -- the same cache the embedding model itself is
    downloaded into, never a plugin-private location.
    """
    local = Path(path_or_repo_id)
    if local.is_dir():
        return local
    import huggingface_hub
    try:
        return Path(huggingface_hub.snapshot_download(
            path_or_repo_id, repo_type="dataset"))
    except Exception:
        return Path(huggingface_hub.snapshot_download(
            path_or_repo_id, repo_type="model"))


def load_prebuilt_store(
    path: Union[str, Path],
    *,
    model_id: str,
    model2vec_version: str,
    strategy=None,
    top_k: int = 3,
    tau: float = 0.1,
    expected_dim: Optional[int] = None,
):
    """Load and verify a prebuilt artifact.

    Returns ``(store_or_None, cache_keys, reason)`` -- see
    ``PrototypeIntentStore.load_prebuilt``.
    """
    from ovos_m2v_pipeline import PrototypeIntentStore
    from ovos_m2v_pipeline.strategies import PrototypeStrategy

    try:
        artifact_dir = resolve_prebuilt_path(str(path))
    except Exception as exc:
        reason = f"could not resolve prebuilt path/repo {path!r}: {exc}"
        LOG.warning(f"prebuilt prototypes: {reason}")
        return None, {}, reason

    manifest_path = artifact_dir / "manifest.json"
    npz_path = artifact_dir / "prototypes.npz"
    if not manifest_path.exists() or not npz_path.exists():
        reason = f"'{artifact_dir}' is missing manifest.json or prototypes.npz"
        LOG.warning(f"prebuilt prototypes: {reason}")
        return None, {}, reason

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        reason = f"unreadable manifest.json in '{artifact_dir}': {exc}"
        LOG.warning(f"prebuilt prototypes: {reason}")
        return None, {}, reason

    if manifest.get("model_id") != model_id:
        reason = (f"model id mismatch (prebuilt for "
                   f"{manifest.get('model_id')!r}, running {model_id!r})")
        LOG.info(f"prebuilt prototypes ignored: {reason}")
        return None, {}, reason
    if manifest.get("model2vec_version") != model2vec_version:
        reason = (f"model2vec version mismatch (prebuilt with "
                   f"{manifest.get('model2vec_version')!r}, running "
                   f"{model2vec_version!r})")
        LOG.info(f"prebuilt prototypes ignored: {reason}")
        return None, {}, reason

    try:
        data = np.load(npz_path, allow_pickle=False)
        embeddings = data["embeddings"].astype(np.float32)
        labels = data["labels"]
    except (OSError, ValueError, KeyError) as exc:
        reason = f"unreadable prototypes.npz in '{artifact_dir}': {exc}"
        LOG.warning(f"prebuilt prototypes: {reason}")
        return None, {}, reason

    dim = embeddings.shape[1] if embeddings.ndim == 2 and len(embeddings) else 0
    manifest_dim = manifest.get("embedding_dim", dim)
    if manifest_dim != dim:
        reason = (f"embedding_dim in manifest ({manifest_dim}) does not "
                   f"match prototypes.npz ({dim})")
        LOG.warning(f"prebuilt prototypes ignored: {reason}")
        return None, {}, reason
    if expected_dim is not None and dim and expected_dim != dim:
        reason = (f"embedding dimension mismatch (prebuilt {dim}, "
                   f"running model {expected_dim})")
        LOG.info(f"prebuilt prototypes ignored: {reason}")
        return None, {}, reason

    strategy = PrototypeStrategy(strategy) if strategy is not None \
        else PrototypeStrategy(manifest.get("prototype_strategy",
                                             PrototypeStrategy.MAX_OVER_ALL.value))
    store = PrototypeIntentStore(
        embeddings if len(embeddings) else None,
        labels if len(labels) else None,
        strategy=strategy, top_k=top_k, tau=tau,
    )
    cache_keys = dict(manifest.get("cache_keys") or {})
    LOG.info(
        f"loaded prebuilt prototype artifact from '{artifact_dir}': "
        f"{len(labels)} vector(s) for {len(manifest.get('labels', []))} label(s)"
    )
    return store, cache_keys, "ok"
