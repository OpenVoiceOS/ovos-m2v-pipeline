import itertools
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
from typing import Any, List, Optional, Union, Dict, Iterable, Tuple

from model2vec.inference import StaticModelPipeline
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch, ConfidenceMatcherPipeline
from ovos_config.locations import get_xdg_data_save_path
from ovos_spec_tools import SpecMessage
from ovos_spec_tools.context import gate_satisfied, context_slot_candidates
from itertools import islice

from ovos_spec_tools.expansion import iter_expand
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG

from ovos_m2v_pipeline.cache import PrototypeCache, compute_cache_key
from ovos_m2v_pipeline.strategies import (
    PrototypeStrategy,
    select_anchors,
    score_labels,
)

#: Regex matching ``{slot}`` placeholders in OVOS-INTENT-1 template samples.
_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

#: retry backoff for a deferred model load that raised: starts at 30s
#: and doubles on each consecutive failure, capped at 15 minutes, so a
#: transient network blip retries soon while a persistently broken
#: model id (bad repo, no network at all) does not re-attempt the load
#: on every single utterance.
_MODEL_LOAD_RETRY_BASE_S = 30.0
_MODEL_LOAD_RETRY_CAP_S = 15 * 60.0

# Labels that bypass the registered-intent check and are always matched
_SPECIAL_LABELS = {"ocp:play", "common_query:common_query", "stop:stop"}

# Map each special label to a substring that must appear in `session.pipeline`
# entries for the label to be considered. Without the matching downstream
# pipeline in the session there is no service to route the intent to.
_SPECIAL_LABEL_PIPELINES = {
    "ocp:play": "ovos-ocp-pipeline-plugin",
    "common_query:common_query": "ovos-common-query-pipeline-plugin",
    "stop:stop": "ovos-stop-pipeline-plugin",
}

# Default ``label_map`` layer: model label -> (skill_id, canonical_label).
# These are the built-in OCP / common-query / stop remaps that predate the
# `label_map` config option; they stay wired unconditionally unless a model
# manifest or user config overrides the same key. Values are ``(skill_id,
# canonical_label)`` tuples (rather than a single ``skill_id:label`` string)
# because the canonical labels here do not live in the `skill_id:intent`
# namespace themselves.
_SPECIAL_LABEL_MAP: Dict[str, Tuple[str, str]] = {
    "ocp:play": ("ovos.common_play", "ovos.common_play.play_search"),
    "common_query:common_query": ("common_query.openvoiceos", "common_query.question"),
    "stop:stop": ("stop.openvoiceos", "mycroft.stop"),
}

#: Built-in per-language default model, keyed by primary lang subtag
#: (e.g. ``"en"`` from ``"en-US"``). Empty: `DEFAULT_MULTILINGUAL` is the
#: default for every language. `OpenVoiceOS/ovos-m2v-intents-en` scores
#: worse than the multilingual model on paraphrase / prototype-mode
#: dispatch ranking (e.g. "lights on now" failing to match prototypes
#: trained on "turn on the lights") despite a comparable held-out
#: accuracy, so it is not wired in as anyone's default; it remains
#: selectable via `config["models"]` for size-constrained deployments
#: that can accept that trade-off.
DEFAULT_MODELS: Dict[str, str] = {}

#: Default model for every language without a `DEFAULT_MODELS` entry
#: (and without a `config["models"]` override) -- i.e. every language.
DEFAULT_MULTILINGUAL = "OpenVoiceOS/ovos-m2v-intents-multilingual"


def _resolve_model_id(config: Dict, lang: str) -> str:
    """Resolve the Model2Vec repo id to load for *lang*.

    Resolution order (highest priority first):

    1. ``config["model"]`` -- an explicit single-model override, unchanged
       from before this per-language roster existed.
    2. ``config["models"]`` -- a ``{locale_or_lang: repo_id}`` map, matched
       first against the full locale (``"pt-BR"``) then against the primary
       subtag (``"pt"``); keys are matched case-insensitively.
    3. `DEFAULT_MODELS`, matched against the primary subtag.
    4. `DEFAULT_MULTILINGUAL`.

    There is no fallback to the deprecated
    ``Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2``
    default: callers that relied on it must now set ``config["model"]``
    explicitly.
    """
    if "model" in config:
        # Explicitly set (including an explicit empty string, which the
        # caller treats as "no model configured" and raises on): never
        # overridden by a per-language default.
        return config["model"]

    lang = (lang or "en-US").lower()
    primary = lang.split("-")[0]

    models_cfg = {str(k).lower(): v for k, v in (config.get("models") or {}).items()}
    if lang in models_cfg:
        return models_cfg[lang]
    if primary in models_cfg:
        return models_cfg[primary]

    if primary in DEFAULT_MODELS:
        return DEFAULT_MODELS[primary]
    return DEFAULT_MULTILINGUAL


def _load_labels_manifest(model_path: str) -> Dict[str, Any]:
    """Load the optional ``labels.json`` a trained model ships alongside it.

    A classifier's label head is frozen at train time, so which bus intents
    its labels denote is a property of the model, not the plugin. When a
    model directory (local path or the local HF hub cache) carries a
    ``labels.json`` manifest, it is used as the *model* layer of the
    ``label_map`` / ``valid_labels`` config, between the built-in defaults
    and the user's own config.

    ``labels.json`` has the same shape as the ``label_map`` config value
    (model label -> ``skill_id:intent`` string), plus an optional
    ``"valid_labels"`` list key.

    This is called from ``__init__``, on the critical path of constructing
    the pipeline, so it must never touch the network: for a hub id it only
    consults the local HF cache (``local_files_only=True``), riding the
    same cache the model's own weights were already fetched into. A model
    not yet cached locally, or one with no manifest, is silently treated as
    having none - it is not a reason to fetch anything here.

    Never raises: a missing, corrupt, or unreadable manifest is logged
    (debug for "not present locally", warning for "present but unreadable")
    and ignored, and matching falls back to the built-in defaults / user
    config.
    """
    try:
        local_dir = Path(model_path)
        if local_dir.exists():
            manifest_path = local_dir / "labels.json"
            if not manifest_path.exists():
                return {}
            raw = manifest_path.read_text(encoding="utf-8")
        else:
            import huggingface_hub

            try:
                cached = huggingface_hub.hf_hub_download(
                    model_path, "labels.json", local_files_only=True
                )
            except (huggingface_hub.errors.LocalEntryNotFoundError,
                     huggingface_hub.errors.EntryNotFoundError,
                     OSError, ValueError):
                # No manifest in the local cache (model not yet cached, no
                # labels.json shipped, or repo id unknown offline) - never
                # reach out to the network to find out; the manifest is
                # optional and, once the model itself is fetched by whatever
                # downloads it, labels.json rides the same cache entry.
                LOG.debug(f"Model2Vec: no local labels.json cached for "
                          f"'{model_path}'")
                return {}
            raw = Path(cached).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError) as e:
        LOG.warning(f"Model2Vec: failed to load labels.json manifest for "
                    f"'{model_path}': {e}")
        return {}
    if not isinstance(data, dict):
        LOG.warning(f"Model2Vec: labels.json manifest for '{model_path}' is "
                    f"not a JSON object, ignoring")
        return {}
    return data


#: Upper bound on entity-filled samples generated per template. The
#: cartesian product over registered entity values is unbounded input
#: (auto-registered .entity files can carry thousands of values each);
#: everything past this bound is a deterministic evenly-strided sample of
#: the combination space.
MAX_ENTITY_EXPANSIONS = 2000


class PrototypeIntentStore:
    """Mutable store of L2-normalised prototype embeddings per intent label.

    Inference is cosine nearest-neighbour: for each label the highest cosine
    similarity across its prototypes is used as the match score.

    The store starts empty and is populated incrementally at runtime as skills
    register their Padatious intents (which carry example utterances).

    Typical usage
    -------------
    Build from lists (offline / testing)::

        store = PrototypeIntentStore()
        store.add(model, "skill:intent", ["example 1", "example 2"], k=5)

    Persist and reload (optional)::

        store.save("prototypes.npz")
        store = PrototypeIntentStore.load("prototypes.npz")
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        *,
        strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        top_k: int = 3,
        tau: float = 0.1,
        cache: Optional[PrototypeCache] = None,
    ) -> None:
        # Bus registration handlers run concurrently on an executor thread
        # pool; this lock keeps the parallel embeddings/labels arrays in sync.
        self._lock = threading.RLock()
        self.strategy: PrototypeStrategy = PrototypeStrategy(strategy)
        self.top_k: int = top_k
        self.tau: float = tau
        #: optional boot-time persistence for add()'s encode step; see
        #: ``ovos_m2v_pipeline.cache.PrototypeCache``. ``None`` disables it.
        self.cache: Optional[PrototypeCache] = cache
        if embeddings is not None and len(embeddings):
            embeddings = np.atleast_2d(embeddings)
            labels_arr = np.asarray(labels, dtype=object)
            if len(embeddings) != len(labels_arr):
                raise ValueError(
                    f"embeddings and labels must have the same length, "
                    f"got {len(embeddings)} vs {len(labels_arr)}"
                )
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self._embeddings: np.ndarray = np.where(
                norms > 0, embeddings / norms, embeddings
            ).astype(np.float32)
            self._labels: np.ndarray = labels_arr
        else:
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            self._labels = np.array([], dtype=object)
        #: chunks appended by add() and stacked lazily: one vstack per add
        #: copies the whole store every registration (quadratic build cost,
        #: measured as a 1.2GB array reallocated per skill on real installs)
        self._pending: List[tuple] = []
        self._label_set = set(np.unique(self._labels)) if len(self._labels) else set()

    def _consolidate(self) -> None:
        """Fold pending chunks into the contiguous arrays.

        ``np.vstack([self._embeddings] + chunks)`` (the previous
        implementation) allocates one brand-new array the size of the
        *entire* resulting store while every existing chunk is still
        alive -- a transient ~2x peak over the final store size. On a
        capped service that single allocation is what stalls: live
        registrations finish (they only append to ``_pending``), and the
        first read afterwards -- ``scores()``, on the bus dispatch thread
        handling the next utterance -- is the one that needs the doubled
        memory and never gets it, wedging dispatch (observed: an ovos-core
        install pinned at its 2G cgroup cap, 100% CPU, no further intents
        matched, 25+ minutes after the last registration completed).

        Instead, allocate the final-size array exactly once and copy each
        source (the old store, then each pending chunk) into it in turn,
        dropping every source as soon as it is copied. Peak transient
        overhead is then bounded by the final array itself plus at most
        one still-alive source (<= ``MAX_ENTITY_EXPANSIONS`` rows), not by
        the size of the whole backlog.

        A ``MemoryError`` here is a hard failure, not something to retry:
        it is logged and the store is left exactly as it was (nothing
        pending is dropped), so a live install keeps matching whatever it
        already consolidated instead of spinning.
        """
        if not self._pending:
            return
        total_new = sum(len(lbls) for _, lbls in self._pending)
        dim = self._pending[0][0].shape[1]
        old_n = len(self._labels)
        try:
            target_emb = np.empty((old_n + total_new, dim), dtype=np.float32)
            target_lbl = np.empty(old_n + total_new, dtype=object)
        except MemoryError:
            LOG.error(
                f"prototype store: out of memory consolidating {total_new} "
                f"pending prototype(s) onto {old_n} existing; keeping the "
                f"store as-is and leaving the batch pending"
            )
            return
        if old_n:
            target_emb[:old_n] = self._embeddings
            target_lbl[:old_n] = self._labels
        self._embeddings = None  # drop the old array before copying chunks in
        self._labels = None
        offset = old_n
        pending, self._pending = self._pending, []
        while pending:
            # pop (not iterate): each source chunk is dereferenced right
            # after its data is copied in, so at most one chunk plus the
            # target array is resident at a time -- never the whole backlog
            chunk, chunk_labels = pending.pop()
            n = len(chunk_labels)
            target_emb[offset:offset + n] = chunk
            target_lbl[offset:offset + n] = chunk_labels
            offset += n
        self._embeddings = target_emb
        self._labels = target_lbl

    # ------------------------------------------------------------------
    # Public read-only views
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> np.ndarray:
        with self._lock:
            self._consolidate()
            return self._embeddings

    @property
    def labels(self) -> np.ndarray:
        with self._lock:
            self._consolidate()
            return self._labels

    @property
    def unique_labels(self) -> np.ndarray:
        return np.unique(self.labels)

    def __len__(self) -> int:
        return len(self._labels) + sum(len(l) for _, l in self._pending)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _add_anchors(self, label: str, anchors: np.ndarray) -> int:
        """Insert already-computed, already-normalised anchor embeddings for
        *label*, replacing any existing prototypes for that label.

        Shared by ``add()``'s normal encode path and its cache-hit path: both
        end up with a set of anchors to store, they only differ in how those
        anchors were obtained (encode + select_anchors vs. a disk read).
        """
        n_added = len(anchors)
        with self._lock:
            if label in self._label_set:
                # re-registration: drop the label's rows from BOTH the
                # consolidated store and any not-yet-folded pending
                # chunk(s), independently of each other. _consolidate()
                # can fail (MemoryError) and leave _pending untouched --
                # if this branch relied on it to clear a label's old
                # pending chunk, a re-registration during/after a failed
                # consolidate would leave the OLD chunk sitting alongside
                # the new one, and scores() (which reads consolidated +
                # pending) would keep matching retired phrasing forever.
                self._consolidate()
                if len(self._labels):
                    mask = self._labels != label
                    self._embeddings = self._embeddings[mask]
                    self._labels = self._labels[mask]
                self._pending = [(c, l) for c, l in self._pending
                                  if not len(l) or l[0] != label]
                self._label_set.discard(label)
            if n_added == 0:
                return 0
            self._pending.append(
                (anchors, np.array([label] * n_added, dtype=object)))
            self._label_set.add(label)
            LOG.info(f"prototype store: +{n_added} prototypes for "
                     f"{label!r} ({len(self)} total, "
                     f"{len(self._label_set)} labels)")
        return n_added

    def add(
        self,
        model,
        label: str,
        sentences: List[str],
        k: Optional[int] = None,
        random_state: int = 42,
        *,
        cache_key: Optional[str] = None,
    ) -> int:
        """Embed *sentences* and add/replace prototypes for *label*.

        ``k`` caps how many anchors the subsampling / clustering strategies
        keep; ``None`` (the default) keeps every sample so exact training
        samples always score a perfect match.

        ``cache_key`` is a hash of this registration's inputs (model id,
        model2vec version, anchor-selection parameters, raw pre-expansion
        samples -- see ``ovos_m2v_pipeline.cache.compute_cache_key``). When
        given and ``self.cache`` is set: a matching cached entry is loaded
        from disk and used in place of encoding (``model.encode()`` is never
        called); a miss falls through to the normal encode path and the
        result is persisted under *cache_key* afterwards. Callers that don't
        care about persistence (tests, offline ``build()``) simply omit it.

        Returns the number of prototypes actually added.
        """
        if not sentences:
            return 0
        if self.cache is not None and cache_key is not None:
            # Prefer the live model's declared output dimension (e.g.
            # model2vec.StaticModel.dim); fall back to whatever dimension
            # the store itself already holds (existing labels or a pending
            # chunk) when the model doesn't expose one. Either way, a
            # cached entry whose dimension disagrees with what is actually
            # in play is a stale entry (e.g. the artifact behind an
            # unchanged model id was retrained in place at a different
            # dimension) -- loading it as a hit would poison the live
            # store with wrong-shape vectors that later crash
            # _consolidate()/scores() for every label, not just this one.
            expected_dim = getattr(model, "dim", None)
            if not isinstance(expected_dim, int):
                expected_dim = None
            if expected_dim is None:
                if len(self._labels):
                    expected_dim = self._embeddings.shape[1]
                elif self._pending:
                    expected_dim = self._pending[-1][0].shape[1]
            cached = self.cache.load(label, cache_key, expected_dim=expected_dim)
            if cached is not None:
                embeddings, _labels = cached
                return self._add_anchors(label, embeddings)
        if len(sentences) > MAX_ENTITY_EXPANSIONS:
            # single choke point: whatever path materialized the samples
            # (entity slot-filling, padatious template expansion, inline
            # payloads), the store never ingests an unbounded batch — the
            # 2000-cap applied only on one expansion path let a real
            # deployment build a million-prototype store and OOM its cgroup
            LOG.warning(f"label {label!r}: {len(sentences)} samples exceed "
                        f"the ingest bound; sampling {MAX_ENTITY_EXPANSIONS} "
                        f"evenly")
            step = (len(sentences) - 1) / (MAX_ENTITY_EXPANSIONS - 1)
            keep = sorted({round(i * step)
                           for i in range(MAX_ENTITY_EXPANSIONS)})
            sentences = [sentences[i] for i in keep]
        # Embed every sample first; the strategy decides which / how many
        # to keep as anchors at storage time. Embedding happens outside the
        # lock: it is the expensive step and touches no shared state.
        # use_multiprocessing=False: model2vec otherwise spawns
        # os.cpu_count() loky workers for batches over its threshold —
        # observed as 24 subprocesses inside a 2G service cgroup, deep swap
        # and an OOM-kill during skill loading. Static-model encoding is
        # in-process numpy; a long-lived service never wants that fan-out.
        embs = np.atleast_2d(
            model.encode(list(sentences),
                         use_multiprocessing=False)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = np.where(norms > 0, embs / norms, embs)
        anchors = select_anchors(
            embs, self.strategy, k=k, random_state=random_state,
        ).astype(np.float32)
        n_added = self._add_anchors(label, anchors)
        if self.cache is not None and cache_key is not None and n_added:
            self.cache.save(
                label, cache_key, anchors,
                np.array([label] * n_added, dtype=object),
            )
        return n_added

    def remove(self, label: str) -> None:
        """Remove all prototypes for *label* (and its on-disk cache entry,
        if caching is enabled -- otherwise a later restart would resurrect
        it from the stale cache)."""
        if self.cache is not None:
            self.cache.remove(label)
        with self._lock:
            if label not in self._label_set:
                return
            self._consolidate()
            if len(self._labels):
                mask = self._labels != label
                self._embeddings = self._embeddings[mask]
                self._labels = self._labels[mask]
            # strip any not-yet-folded pending chunk(s) too: _consolidate()
            # may have failed (MemoryError) and left _pending untouched
            self._pending = [(c, l) for c, l in self._pending
                              if not len(l) or l[0] != label]
            self._label_set.discard(label)

    def remove_skill(self, skill_id: str) -> None:
        """Remove all prototypes whose label starts with ``<skill_id>:``
        (and their on-disk cache entries, if caching is enabled)."""
        if self.cache is not None:
            self.cache.remove_skill(skill_id)
        with self._lock:
            if not len(self):
                return
            self._consolidate()
            prefix = skill_id + ":"
            if len(self._labels):
                mask = np.array([not str(lbl).startswith(prefix) for lbl in self._labels])
                self._embeddings = self._embeddings[mask]
                self._labels = self._labels[mask]
            # strip any not-yet-folded pending chunk(s) too: _consolidate()
            # may have failed (MemoryError) and left _pending untouched
            self._pending = [(c, l) for c, l in self._pending
                              if not len(l) or not str(l[0]).startswith(prefix)]
            remaining = (set(np.unique(self._labels)) if len(self._labels) else set())
            remaining |= {l[0] for _, l in self._pending if len(l)}
            self._label_set = remaining

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def scores(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """Return the max cosine similarity per label.

        Deliberately does *not* call ``_consolidate()``: this runs on the
        bus dispatch thread for every live utterance, and a label's
        prototypes are always fully contained in either the consolidated
        store or exactly one pending chunk (``add()``/``remove()`` fold a
        label's old rows away before a re-registration appends its new
        ones), so scoring each pending chunk separately and merging the
        per-label results is equivalent to scoring the consolidated whole
        -- without ever needing the consolidation copy. Forcing that copy
        here is what wedged live matching behind a registration burst: the
        first utterance after a burst paid for the whole backlog's
        transient allocation on the dispatch thread.

        Parameters
        ----------
        query_embedding:
            Raw (unnormalised) embedding vector of shape ``(dim,)``.
        """
        with self._lock:
            if not len(self):
                return {}
            sources = [(self._embeddings, self._labels)] if len(self._labels) else []
            sources += list(self._pending)
        q = query_embedding.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        out: Dict[str, float] = {}
        for embeddings, labels in sources:
            for lbl, score in score_labels(
                q, embeddings, labels,
                self.strategy, top_k=self.top_k, tau=self.tau,
            ).items():
                if lbl not in out or score > out[lbl]:
                    out[lbl] = score
        return out

    # ------------------------------------------------------------------
    # Bulk construction helpers (offline / testing)
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model,
        sentences: List[str],
        labels: List[str],
        k: Optional[int] = None,
        random_state: int = 42,
        *,
        strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        top_k: int = 3,
        tau: float = 0.1,
    ) -> "PrototypeIntentStore":
        """Build a store from parallel *sentences* / *labels* lists."""
        store = cls(strategy=strategy, top_k=top_k, tau=tau)
        label_to_sentences: Dict[str, List[str]] = {}
        for sent, lbl in zip(sentences, labels):
            label_to_sentences.setdefault(lbl, []).append(sent)
        for lbl, sents in label_to_sentences.items():
            store.add(model, lbl, sents, k=k, random_state=random_state)
        return store

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save to a NumPy ``.npz`` archive."""
        with self._lock:
            self._consolidate()
            np.savez(path, embeddings=self._embeddings, labels=self._labels.astype(str))
        LOG.info(
            f"Saved {len(self)} prototypes "
            f"for {len(self.unique_labels)} labels -> {path}"
        )

    @classmethod
    def load(
        cls,
        path: str,
        *,
        strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        top_k: int = 3,
        tau: float = 0.1,
    ) -> "PrototypeIntentStore":
        """Load from a NumPy ``.npz`` archive."""
        data = np.load(path, allow_pickle=False)
        return cls(
            data["embeddings"].astype(np.float32), data["labels"],
            strategy=strategy, top_k=top_k, tau=tau,
        )


def _parse_intent_file(path: str, ctx: str = "") -> List[str]:
    """Return expanded, non-empty, non-comment lines from a Padatious ``.intent`` file.

    Template syntax (``(a|b)``, ``[optional]``) is expanded via
    ``ovos_spec_tools.expansion.expand`` so that every concrete
    utterance variant is represented as a separate prototype.

    A line that is not parsable as OVOS-INTENT-1 grammar is logged and
    skipped so that a single bad template never discards the rest of the
    file (OVOS-INTENT-4 §6.3); *ctx* carries the §5.3 identifier fields
    for those warnings.
    """
    try:
        sentences: List[str] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.lstrip().startswith("#"):
                    try:
                        # lazy + bounded: a combinatorial template must
                        # not materialize its full product just to be
                        # sampled down at store ingest
                        sentences.extend(islice(iter_expand(line),
                                                MAX_ENTITY_EXPANSIONS))
                    except Exception as exc:
                        LOG.warning(f"skipping malformed template {line!r}: "
                                    f"{exc} {ctx}")
        return sentences
    except OSError as exc:
        LOG.warning(f"Could not read intent file '{path}': {exc}")
        return []


def _raw_intent_lines(path: str) -> List[str]:
    """Return the raw, pre-expansion non-comment lines of a Padatious
    ``.intent`` file, for prototype-cache key hashing (see
    ``ovos_m2v_pipeline.cache.compute_cache_key``).

    Unlike ``_parse_intent_file`` this never expands OVOS-INTENT-1 template
    syntax: the cache key is over the raw templates, not their expansion. A
    read failure yields an empty list -- callers treat that as "skip caching
    for this registration", never as a reason to reject it.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return [line.strip() for line in fh
                    if line.strip() and not line.strip().startswith("#")]
    except OSError:
        return []


class Model2VecIntentPipeline(ConfidenceMatcherPipeline):
    """A pipeline that integrates Model2Vec with OVOS for intent matching.

    Two operating modes are supported via ``config["mode"]``:

    ``"classifier"`` (default)
        Loads a ``StaticModelPipeline`` (embedding model + trained linear
        classifier head).  Scores are softmax probabilities.

    ``"prototype"``
        Loads a bare ``StaticModel`` (embeddings only, **no training step**).
        At runtime, builds a ``PrototypeIntentStore`` from the example
        utterances supplied by Padatious intent registrations.  Scores are
        cosine similarities.

        Padatious ``.intent`` files are read directly when
        ``padatious:register_intent`` fires; the file path is taken from
        ``message.data["file_name"]``.  Adapt intents are tracked by label
        name only and are not matched in prototype mode.

    Configuration keys (prototype mode)
    ------------------------------------
    ``prototype_k`` : int, optional (default: unlimited)
        Maximum number of prototype embeddings kept per label (used by the
        strategies that subsample / cluster — see ``prototype_strategy``).
        Unset (the default) keeps every registered sample as an anchor so
        that an exact training sample always scores a perfect match;
        set an integer to cap memory at the cost of recall.
    ``prototype_strategy`` : str, default ``"max_over_all"``
        One of the ``PrototypeStrategy`` values
        (``mean_centroid`` / ``medoid`` / ``max_over_all`` / ``top_k_mean`` /
        ``farthest_point`` / ``kmeans_centers`` / ``softmax_weighted``).
    ``prototype_top_k`` : int, default 3
        K for ``top_k_mean``.
    ``prototype_tau`` : float, default 0.1
        Softmax temperature for ``softmax_weighted``.
    """

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        config = config or Configuration().get('intents', {}).get("ovos_m2v_pipeline") or dict()
        super().__init__(bus, config)
        lang = Configuration().get("lang", "en-us")
        model_path = _resolve_model_id(self.config, lang)
        if not model_path:
            raise FileNotFoundError("'model' not set in configuration for ovos_m2v_pipeline")

        self.intents: set = set()
        self.ignore_labels: List[str] = self.config.get("ignore_intents") or []
        #: Merged `label_map` layers: built-in defaults < model manifest
        #: (`labels.json`, when the loaded model ships one) < user config.
        #: Consumed by `_apply_special_label_map` to turn a raw model label
        #: into `(skill_id, canonical_label)`.
        manifest = _load_labels_manifest(model_path)
        manifest_map = {k: v for k, v in manifest.items() if k != "valid_labels"}
        user_map = self.config.get("label_map") or {}
        self.label_map: Dict[str, Any] = {**_SPECIAL_LABEL_MAP, **manifest_map, **user_map}
        #: Allow-list of raw model labels eligible to match in classifier mode
        #: (`_match_classifier`), checked BEFORE `label_map` resolution
        #: (pre-mapping): a manifest's `valid_labels` describes the model's
        #: label vocabulary, not the post-map bus topics. `None` disables the
        #: allow-list check. Not consulted in prototype mode
        #: (`_match_prototype`): the prototype store is itself the allow-list,
        #: since it only ever holds runtime-registered labels.
        self.valid_labels: Optional[List[str]] = self.config.get("valid_labels")
        if self.valid_labels is None:
            manifest_valid = manifest.get("valid_labels")
            if isinstance(manifest_valid, list):
                self.valid_labels = manifest_valid
        self._syncing = False
        #: Registered entity value-sets (OVOS-INTENT-4 §7), keyed by entity
        #: name (lowercase). Used to fill ``{slot}`` placeholders in template
        #: samples before embedding. Disabled (left empty) in classifier mode.
        self.entities: Dict[str, List[str]] = {}
        #: OVOS-CONTEXT-1 §6/§6.1 gating declarations per registered label,
        #: keyed by label -> (requires_context, excludes_context). Each list
        #: holds bare-string keys or ``{"key", "scope"}`` mappings and is
        #: evaluated at match time via ``gate_satisfied``.
        self._context_gates: Dict[str, Tuple[list, list]] = {}
        #: Per-label suppression phrases (OVOS-INTENT-4 §6.1 ``blacklist``),
        #: keyed by intent label. A candidate is dropped at match time when the
        #: utterance contains one of its label's blacklisted phrases. Named and
        #: matched consistently with the padacioso engine's ``excluded_keywords``.
        self.excluded_keywords: Dict[str, List[str]] = {}
        #: Declared slot names per registered label (OVOS-CONTEXT-1 §7),
        #: parsed from the label's original template samples before entity
        #: expansion rewrites ``{slot}`` placeholders into concrete values.
        #: Used to fill declared slots from live intent context independently
        #: of ``requires_context`` (which gates the match, not the fill).
        self._intent_slots: Dict[str, List[str]] = {}
        #: skill_ids that already triggered the frozen-classifier INTENT-4
        #: warning (see ``_handle_intent4_register_template``), so the
        #: warning logs once per skill rather than once per template.
        self._intent4_frozen_warned: set = set()
        #: model labels for which the colon-less `label_map` target warning
        #: has already been logged, so it logs once per label rather than
        #: once per match (mirrors `_intent4_frozen_warned`).
        self._label_map_warned: set = set()

        mode = self.config.get("mode", "classifier")
        self._mode = mode
        self._model_path = model_path
        #: guards `self.model` (assigned exactly once, by `_load_model_now`)
        #: and `self._pending_additions` (mutated from bus-handler threads
        #: before the model exists, and drained once it does).
        self._model_lock = threading.RLock()
        self.model = None
        self._model_load_thread: Optional[threading.Thread] = None
        self._warmup_logged = False
        #: consecutive failed load attempts, and the monotonic time (if any)
        #: before which a new attempt should not be started -- see
        #: `_MODEL_LOAD_RETRY_BASE_S` / `_MODEL_LOAD_RETRY_CAP_S`.
        self._model_load_failures: int = 0
        self._model_load_retry_at: Optional[float] = None
        #: registrations that arrived before the model finished loading:
        #: (label, sentences, k, cache_key) tuples, encoded exactly once by
        #: `_flush_pending_additions` when the deferred load completes.
        self._pending_additions: List[Tuple[str, List[str], Optional[int], Optional[str]]] = []
        #: how long a match call waits for a cold-start model load before
        #: giving up on THIS utterance and warming up in the background.
        self._model_load_budget: float = float(self.config.get("model_load_budget", 0.5))
        preload = bool(self.config.get("preload_model", False))

        if mode == "prototype":
            self._prototype_k: Optional[int] = self.config.get("prototype_k")
            self._prototype_strategy: PrototypeStrategy = PrototypeStrategy(
                self.config.get("prototype_strategy",
                                PrototypeStrategy.MAX_OVER_ALL.value)
            )
            self._prototype_top_k: int = self.config.get("prototype_top_k", 3)
            self._prototype_tau: float = self.config.get("prototype_tau", 0.1)

            #: boot-time persistence for add()'s encode step (see
            #: ovos_m2v_pipeline.cache): each registration's inputs (model
            #: id, model2vec version, anchor-selection params, raw samples)
            #: hash to a cache key, so an unchanged registration across
            #: restarts loads its embeddings from disk instead of
            #: re-encoding. `prototype_cache: false` disables it outright.
            self._model_id = model_path
            import model2vec
            self._model2vec_version = getattr(model2vec, "__version__", "")
            self._prototype_cache_enabled: bool = bool(
                self.config.get("prototype_cache", True))
            cache = None
            if self._prototype_cache_enabled:
                cache_dir = self.config.get("prototype_cache_dir") or str(
                    Path(get_xdg_data_save_path()) / "m2v_prototypes")
                cache = PrototypeCache(Path(cache_dir))

            self.prototype_store: Optional[PrototypeIntentStore] = PrototypeIntentStore(
                strategy=self._prototype_strategy,
                top_k=self._prototype_top_k,
                tau=self._prototype_tau,
                cache=cache,
            )

            self.bus.on("mycroft.ready", self._handle_ready_prototype)
            # Legacy registration topics (kept alongside OVOS-INTENT-4).
            self.bus.on("padatious:register_intent", self._handle_register_padatious)
            self.bus.on("register_intent", self._handle_register_adapt)
            self.bus.on("detach_intent", self._handle_detach_intent)
            self.bus.on("detach_skill", self._handle_detach_skill)
            # OVOS-INTENT-4 registration topics. m2v matches on example
            # utterances, so it is a TEMPLATE-style engine: it consumes
            # `ovos.intent.register.template` (§6) and NOT
            # `ovos.intent.register.keyword` (§11).
            self._wire_intent4_handlers()

            LOG.info(
                f"Registered Model2VecIntents pipeline (prototype mode) "
                f"with model: '{model_path}' (model load deferred until first use)"
            )
        else:
            self.prototype_store = None

            # Register event handlers for intent synchronization
            self.bus.on("mycroft.ready", self.handle_sync_intents)
            self.bus.on("padatious:register_intent", self.handle_sync_intents)
            self.bus.on("register_intent", self.handle_sync_intents)
            self.bus.on("detach_intent", self.handle_sync_intents)
            self.bus.on("detach_skill", self.handle_sync_intents)
            # OVOS-INTENT-4 registration topics. In classifier mode the model
            # is frozen, so registrations only (de)gate which trained classes
            # are eligible — exactly like the legacy manifest sync above.
            self._wire_intent4_handlers()

            LOG.info(
                f"Registered Model2VecIntents pipeline with model: '{model_path}' "
                f"(model load deferred until first use)"
            )

            # Seed the intent allowlist from skills loaded before this pipeline.
            # Bus events keep it in sync for skills that load/unload later.
            self._initial_intent_sync()

        if preload:
            self._ensure_model(background_ok=False)

    def _load_model_now(self) -> None:
        """Load ``self.model`` and drain any buffered prototype registrations.

        Runs either synchronously (``preload_model: true``, or a caller with
        ``background_ok=False``) or on a background thread kicked off by
        ``_ensure_model``; either way at most one load is in flight at a
        time thanks to ``_model_load_thread`` being created/cleared under
        ``_model_lock``.

        On failure, clears ``_model_load_thread`` so a later ``_ensure_model``
        call can start a fresh attempt (never retried automatically here --
        a broken model id must not spin the loader on every utterance) and
        schedules that retry no sooner than the current backoff window
        (``_MODEL_LOAD_RETRY_BASE_S``, doubling, capped at
        ``_MODEL_LOAD_RETRY_CAP_S``). A dev-eager-load-at-construction repo
        raised loudly at boot and stayed dead until a restart fixed the
        config; this instead keeps retrying on its own once the underlying
        cause (e.g. a network blip) clears.
        """
        start = time.monotonic()
        try:
            if self._mode == "prototype":
                from model2vec import StaticModel
                model = StaticModel.from_pretrained(self._model_path)
            else:
                model = StaticModelPipeline.from_pretrained(self._model_path)
        except Exception:
            with self._model_lock:
                self._model_load_failures += 1
                backoff = min(
                    _MODEL_LOAD_RETRY_BASE_S * (2 ** (self._model_load_failures - 1)),
                    _MODEL_LOAD_RETRY_CAP_S,
                )
                self._model_load_retry_at = time.monotonic() + backoff
                self._model_load_thread = None
                self._warmup_logged = False
            LOG.exception(f"failed to load deferred Model2Vec model "
                          f"'{self._model_path}' (attempt {self._model_load_failures}); "
                          f"retrying in {backoff:.0f}s")
            return
        with self._model_lock:
            self.model = model
            self._model_load_failures = 0
            self._model_load_retry_at = None
            if self.prototype_store is not None:
                self._flush_pending_additions()
        LOG.info(f"Model2Vec model '{self._model_path}' loaded in "
                 f"{time.monotonic() - start:.2f}s (deferred load)")

    def _ensure_model(self, background_ok: bool = True) -> bool:
        """Make sure ``self.model`` is loaded, returning ``True`` once it is.

        The first caller (construction with ``preload_model``, or the first
        match) starts the load thread; every other caller just waits on it.
        When ``background_ok`` and the load has not finished within
        ``model_load_budget`` seconds, returns ``False`` so the caller can
        skip this one utterance instead of blocking the bus thread
        indefinitely — the load keeps running in the background and later
        calls pick up the now-ready model.

        A load that previously failed is not retried on every call: while
        inside the current backoff window (see ``_load_model_now``) this
        returns ``False`` immediately without spawning a new thread; once
        the window elapses, the next caller starts a fresh attempt.
        """
        if self.model is not None:
            return True
        if not hasattr(self, "_model_lock"):
            # object built via `__new__`, bypassing `__init__` (white-box
            # tests): there is nothing to defer, `self.model` is simply unset
            return False
        with self._model_lock:
            if self.model is not None:
                return True
            if self._model_load_thread is None:
                retry_at = self._model_load_retry_at
                if retry_at is not None and time.monotonic() < retry_at:
                    # still backing off after a previous failure
                    return False
                self._model_load_thread = threading.Thread(
                    target=self._load_model_now,
                    name="m2v-deferred-model-load", daemon=True)
                self._model_load_thread.start()
            thread = self._model_load_thread
        if not background_ok:
            thread.join()
            return self.model is not None
        thread.join(timeout=self._model_load_budget)
        if self.model is None:
            if not self._warmup_logged:
                self._warmup_logged = True
                LOG.info("Model2Vec model is still warming up; skipping this "
                         "utterance, matching resumes once it is ready")
            return False
        return True

    def _flush_pending_additions(self) -> None:
        """Encode every registration buffered while the model was loading.

        Must be called with ``self._model_lock`` held and ``self.model`` set
        (see ``_load_model_now``); each entry is encoded exactly once.
        """
        pending, self._pending_additions = self._pending_additions, []
        for label, sentences, k, cache_key in pending:
            try:
                self.prototype_store.add(self.model, label, sentences, k=k,
                                         cache_key=cache_key)
            except Exception as exc:
                LOG.error(f"deferred prototype add failed for '{label}': {exc}")

    def _add_prototypes(self, label: str, sentences: List[str],
                        k: Optional[int], cache_key: Optional[str]) -> int:
        """Encode *sentences* now if the model is loaded, otherwise buffer
        the raw registration for ``_flush_pending_additions`` to encode once
        the deferred load completes."""
        # `_model_lock` is only absent for objects built via `__new__` that
        # skip `__init__` entirely (a handful of white-box tests) -- those
        # already hand-set `self.model`, so fall back to the pre-deferred
        # encode-now behaviour rather than crash.
        lock = getattr(self, "_model_lock", None)
        if lock is None:
            return self.prototype_store.add(self.model, label, sentences, k=k,
                                            cache_key=cache_key)
        with lock:
            if self.model is None:
                self._pending_additions.append((label, list(sentences), k, cache_key))
                return len(sentences)
            return self.prototype_store.add(self.model, label, sentences, k=k,
                                            cache_key=cache_key)

    def _forget_pending(self, label: str) -> None:
        """Drop any buffered-but-unencoded registration for *label* (mirrors
        ``prototype_store.remove`` for entries that never reached the store)."""
        pending = getattr(self, "_pending_additions", None)
        if pending:
            self._pending_additions = [p for p in pending if p[0] != label]

    def _forget_pending_skill(self, skill_id: str) -> None:
        """Drop buffered-but-unencoded registrations owned by *skill_id*
        (mirrors ``prototype_store.remove_skill``)."""
        pending = getattr(self, "_pending_additions", None)
        if pending:
            prefix = skill_id + ":"
            self._pending_additions = [p for p in pending
                                       if not p[0].startswith(prefix)]

    def _initial_intent_sync(self) -> None:
        """Query adapt + padatious manifests once at startup.

        Skills loaded *before* this pipeline never emit `register_intent`, so
        without this pull they would stay invisible until the next
        `mycroft.ready` / register / detach event.
        """
        timeout = self.config.get("timeout", 1)
        adapt: List[str] = []
        padatious: List[str] = []
        try:
            adapt = self._get_adapt_intents(timeout)
        except RuntimeError:
            LOG.debug("Model2Vec: adapt manifest not available at startup")
        try:
            padatious = self._get_padatious_intents(timeout)
        except RuntimeError:
            LOG.debug("Model2Vec: padatious manifest not available at startup")
        if adapt or padatious:
            self.intents = list(set(adapt + padatious))
            LOG.debug(f"Model2Vec seeded {len(self.intents)} intents on startup")

    # ------------------------------------------------------------------
    # Classifier-mode intent synchronisation
    # ------------------------------------------------------------------

    def _get_adapt_intents(self, timeout: int = 1) -> List[str]:
        """
        Retrieves Adapt intent names from the message bus, excluding ignored labels.

        Args:
            timeout: Maximum time in seconds to wait for the response.

        Returns:
            A list of Adapt intent names excluding ignored labels.

        Raises:
            RuntimeError: If no response is received from the bus.
        """
        msg = Message("intent.service.adapt.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.adapt.manifest", timeout=timeout)
        if not res:
            raise RuntimeError("Failed to retrieve intent names")
        return [i["name"] for i in res.data["intents"] if i["name"] not in self.ignore_labels]

    def _get_padatious_intents(self, timeout: int = 1) -> List[str]:
        """
        Retrieves Padatious intent names from the message bus, excluding ignored labels.

        Args:
            timeout: Maximum time in seconds to wait for the response.

        Returns:
            A list of Padatious intent names not present in the ignore list.

        Raises:
            RuntimeError: If no response is received from the bus.
        """
        msg = Message("intent.service.padatious.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.padatious.manifest", timeout=timeout)
        if not res:
            raise RuntimeError("Failed to retrieve intent names")
        return [i for i in res.data["intents"] if i not in self.ignore_labels]

    def handle_sync_intents(self, message: Message) -> None:
        """
        Synchronizes registered intents when new skills are loaded or existing ones are detached.

        Args:
            message: The message that triggered intent synchronization.
        """
        # Sync newly (de)registered intents with debounce
        if self._syncing:
            return
        self._syncing = True
        try:
            time.sleep(3)
            timeout = self.config.get("timeout", 1)
            self.intents = set(
                self._get_adapt_intents(timeout) + self._get_padatious_intents(timeout)
            )
            LOG.debug(f"Model2Vec registered intents: {len(self.intents)}")
        except RuntimeError:
            pass
        finally:
            self._syncing = False

    def _allowed_special_labels(self, message: Optional[Message]) -> set:
        """Return the special labels enabled by the caller's session pipeline.

        `ocp:play`, `common_query:common_query`, and `stop:stop` are only
        meaningful when their respective downstream pipelines are present in
        ``session.pipeline``.  If we cannot determine the session (no message
        or no session in context) we fall back to all special labels so the
        plugin keeps working in headless / test contexts.
        """
        if message is None:
            return set(_SPECIAL_LABELS)
        try:
            sess = SessionManager.get(message)
            sess_pipeline = list(sess.pipeline or [])
        except Exception:
            return set(_SPECIAL_LABELS)
        if not sess_pipeline:
            return set(_SPECIAL_LABELS)
        allowed = set()
        for label, needle in _SPECIAL_LABEL_PIPELINES.items():
            if any(needle in p for p in sess_pipeline):
                allowed.add(label)
        return allowed

    # ------------------------------------------------------------------
    # Prototype-mode intent registration
    # ------------------------------------------------------------------

    def _prototype_cache_key(
        self, raw_samples: List[str],
        entity_values: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[str]:
        """Hash a registration's inputs into a prototype-cache key, or
        ``None`` when caching is disabled / there is nothing to hash.

        Never raises: a hashing failure just disables caching for this one
        registration (``add()`` still runs the normal encode path), it is
        never a reason to reject the registration itself.
        """
        if not getattr(self, "_prototype_cache_enabled", False) or not raw_samples:
            return None
        try:
            return compute_cache_key(
                self._model_id, self._model2vec_version,
                {"k": self._prototype_k,
                 "strategy": self._prototype_strategy.value,
                 "max_expansions": MAX_ENTITY_EXPANSIONS},
                raw_samples, entity_values,
            )
        except Exception as exc:
            LOG.warning(f"prototype cache: failed to compute cache key, "
                        f"registration will re-encode: {exc}")
            return None

    def _handle_ready_prototype(self, message: Message) -> None:
        LOG.info(
            f"Model2Vec prototype store ready: {len(self.prototype_store)} prototypes "
            f"for {len(self.prototype_store.unique_labels)} labels"
        )

    def _handle_register_padatious(self, message: Message) -> None:
        """Build prototypes from a Padatious ``.intent`` file or inline samples.

        ``message.data["samples"]`` (pre-expanded list) is preferred when
        present; otherwise the file at ``message.data["file_name"]`` is read
        and its template syntax is expanded lazily via ``iter_expand``.
        """
        name: str = message.data.get("name", "")
        if name.endswith(".intent"):
            # ovos-workshop dual-registers one intent on both wire contracts
            # (legacy name suffixed ".intent", spec name suffixless); fold to
            # one canonical label or the store holds every prototype twice
            name = name[:-len(".intent")]
        # ignore_labels entries may be written in either wire form
        if not name or name in self.ignore_labels                 or f"{name}.intent" in self.ignore_labels:
            return

        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        ctx = (f"[skill_id={skill_id!r} name={name!r} "
               f"lang={message.data.get('lang')!r} topic={message.msg_type}]")

        inline = message.data.get("samples") or []
        if inline:
            raw_samples: List[str] = list(inline)
            sentences: List[str] = []
            for s in inline:
                try:
                    sentences.extend(islice(iter_expand(s),
                                            MAX_ENTITY_EXPANSIONS))
                except Exception as exc:
                    LOG.warning(f"skipping malformed template {s!r}: {exc} {ctx}")
        else:
            file_name: str = message.data.get("file_name", "")
            raw_samples = _raw_intent_lines(file_name) if file_name else []
            sentences = _parse_intent_file(file_name, ctx) if file_name else []

        if not sentences:
            # zero valid templates -> the whole registration is malformed
            LOG.warning(f"rejecting registration: no valid template remains {ctx}")
            return
        cache_key = self._prototype_cache_key(raw_samples)
        try:
            n = self._add_prototypes(name, sentences, self._prototype_k, cache_key)
        except Exception as exc:
            LOG.error(f"Failed to add prototypes for Padatious intent "
                      f"'{name}': {exc}")
            return
        self.intents.add(name)
        self._store_context_gate(name, message)
        LOG.debug(f"Prototype store: added {n} prototype(s) for '{name}'")

    def _handle_register_adapt(self, message: Message) -> None:
        """Track Adapt intent labels (no example sentences -> no prototypes)."""
        name: str = message.data.get("name", "")
        if name and name not in self.ignore_labels:
            self.intents.add(name)

    def _handle_detach_intent(self, message: Message) -> None:
        name: str = message.data.get("intent_name", "")
        if name.endswith(".intent"):
            # mirror the registration-side dealiasing
            name = name[:-len(".intent")]
        if name:
            self.prototype_store.remove(name)
            self._forget_pending(name)
            self.intents.discard(name)
            self._context_gates.pop(name, None)
            self._intent_slots.pop(name, None)
            LOG.debug(f"Prototype store: removed prototypes for '{name}'")

    def _handle_detach_skill(self, message: Message) -> None:
        skill_id: str = message.data.get("skill_id") or message.context.get("skill_id", "")
        if skill_id:
            self.prototype_store.remove_skill(skill_id)
            self._forget_pending_skill(skill_id)
            self.intents = {i for i in self.intents if not i.startswith(skill_id + ":")}
            self._context_gates = {l: g for l, g in self._context_gates.items()
                                   if not l.startswith(skill_id + ":")}
            self._intent_slots = {l: s for l, s in self._intent_slots.items()
                                  if not l.startswith(skill_id + ":")}
            LOG.debug(f"Prototype store: removed prototypes for skill '{skill_id}'")

    # ------------------------------------------------------------------
    # OVOS-INTENT-4 registration (template method) — consumed in addition
    # to the legacy topics. m2v is a template-style engine (it matches on
    # example utterances), so it subscribes to `ovos.intent.register.template`
    # (§6) and `ovos.entity.register` (§7), plus the shared deregister /
    # enable / disable topics (§8). It deliberately does NOT consume
    # `ovos.intent.register.keyword` (§11 — keyword engines only).
    # ------------------------------------------------------------------

    def _wire_intent4_handlers(self) -> None:
        """Subscribe to the OVOS-INTENT-4 registration topics (§4)."""
        self.bus.on(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                    self._handle_intent4_register_template)
        self.bus.on(SpecMessage.ENTITY_REGISTER.value,
                    self._handle_intent4_register_entity)
        self.bus.on(SpecMessage.INTENT_DEREGISTER.value,
                    self._handle_intent4_deregister_intent)
        self.bus.on(SpecMessage.ENTITY_DEREGISTER.value,
                    self._handle_intent4_deregister_entity)
        self.bus.on(SpecMessage.SKILL_DEREGISTER.value,
                    self._handle_intent4_deregister_skill)
        self.bus.on(SpecMessage.INTENT_ENABLE.value,
                    self._handle_intent4_enable)
        self.bus.on(SpecMessage.INTENT_DISABLE.value,
                    self._handle_intent4_disable)

    @staticmethod
    def _intent4_label(message: Message) -> str:
        """Build the internal ``<skill_id>:<intent_name>`` label from §3.2 fields."""
        skill_id = message.data.get("skill_id") or message.context.get("skill_id", "")
        intent_name = message.data.get("intent_name", "")
        if not skill_id or not intent_name:
            return ""
        return f"{skill_id}:{intent_name}"

    def _intent4_warn(self, topic: str, message: Message, reason: str) -> None:
        """Log a malformed-registration rejection at WARN (§5.3 / §6.3 / §7.2)."""
        LOG.warning(
            f"rejecting {topic} registration: {reason} "
            f"[skill_id={message.data.get('skill_id')!r} "
            f"name={message.data.get('intent_name') or message.data.get('entity_name')!r} "
            f"lang={message.data.get('lang')!r}]"
        )

    def _expand_entities(self, samples: List[str]) -> List[str]:
        """Fill ``{slot}`` placeholders in template *samples* with registered
        entity values (OVOS-INTENT-4 §7). Samples without placeholders, or whose
        entity is unregistered, are passed through with the placeholder left
        literal (entities are an optional hint, §7).
        """
        if not self.entities:
            return list(samples)
        out: List[str] = []
        for tmpl in samples:
            slots = _SLOT_RE.findall(tmpl)
            if not slots:
                out.append(tmpl)
                continue
            slot_values: List[List[str]] = []
            for slot in slots:
                vals = self.entities.get(slot.lower())
                slot_values.append(vals if vals else ["{" + slot + "}"])
            # the cartesian product over large value sets explodes: two
            # ~2000-value entities in one template is ~4M strings, all
            # materialized and embedded on every registration — enough to
            # swap out and OOM-kill a capped service. Engines own bounding
            # unbounded entity data: take a deterministic, evenly-strided
            # sample of the combination space instead.
            sizes = [len(v) for v in slot_values]
            total = 1
            for n in sizes:
                total *= n
            if total > MAX_ENTITY_EXPANSIONS:
                LOG.warning(
                    f"template {tmpl!r} expands to {total} combinations; "
                    f"sampling {MAX_ENTITY_EXPANSIONS} evenly")
                step = (total - 1) / (MAX_ENTITY_EXPANSIONS - 1)
                indices = {round(i * step) for i in range(MAX_ENTITY_EXPANSIONS)}
                combos = []
                for idx in sorted(indices):
                    combo = []
                    rem = idx
                    for n in reversed(sizes):
                        combo.append(rem % n)
                        rem //= n
                    combos.append(tuple(slot_values[d][c] for d, c in
                                        enumerate(reversed(combo))))
            else:
                combos = itertools.product(*slot_values)
            for combo in combos:
                filled = tmpl
                for slot, val in zip(slots, combo):
                    filled = filled.replace("{" + slot + "}", val, 1)
                out.append(filled.strip())
        return out

    def _handle_intent4_register_template(self, message: Message) -> None:
        """Register a template intent (§6): bracket-expand + entity-fill the
        ``samples`` and embed them as prototypes (prototype mode) or track the
        label name (classifier mode)."""
        topic = SpecMessage.INTENT_REGISTER_TEMPLATE.value
        label = self._intent4_label(message)
        if not label:
            self._intent4_warn(topic, message, "missing skill_id or intent_name")
            return
        if label in self.ignore_labels:
            return
        samples = message.data.get("samples")
        if not samples:  # missing or empty -> malformed (§6.3)
            self._intent4_warn(topic, message, "samples missing or empty")
            return

        blacklist = message.data.get("blacklist")
        if blacklist:  # §6.1 suppression phrases: drop matches containing these
            self.excluded_keywords[label] = list(blacklist)

        # Classifier mode is frozen: only gate the (already trained) label.
        if self.prototype_store is None:
            skill_id = message.data.get("skill_id") or message.context.get("skill_id", "")
            if skill_id and skill_id not in self._intent4_frozen_warned:
                self._intent4_frozen_warned.add(skill_id)
                LOG.warning(
                    "Model2VecIntentPipeline is a frozen classifier and cannot "
                    "match registered skill intents; use "
                    "ovos-m2v-prototype-pipeline-* to match INTENT-4 "
                    f"registrations from {skill_id}"
                )
            self.intents.add(label)
            self._store_context_gate(label, message)
            LOG.debug(f"Model2Vec: tracking INTENT-4 template label '{label}'")
            return

        expanded: List[str] = []
        for s in self._expand_entities(list(samples)):
            if len(expanded) >= MAX_ENTITY_EXPANSIONS:
                break
            try:
                expanded.extend(islice(
                    iter_expand(s),
                    MAX_ENTITY_EXPANSIONS - len(expanded)))
            except Exception as exc:
                # skip the unparsable template, keep the valid ones (§6.3)
                LOG.warning(
                    f"skipping malformed template {s!r}: {exc} "
                    f"[skill_id={message.data.get('skill_id')!r} "
                    f"name={message.data.get('intent_name')!r} "
                    f"lang={message.data.get('lang')!r} topic={topic}]")
        expanded = [s for s in (e.strip() for e in expanded) if s]
        if not expanded:  # zero non-empty expansions -> malformed (§6.3)
            self._intent4_warn(topic, message, "samples expand to zero non-empty templates")
            return
        # entity values referenced by this template's slots: part of the
        # cache key because they feed `_expand_entities` above, but they
        # come from a `set` upstream (`_handle_intent4_register_entity`) so
        # their list order is unstable across runs -- compute_cache_key()
        # sorts them before hashing.
        slots = {slot for s in samples for slot in _SLOT_RE.findall(s)}
        entity_values = {slot: self.entities[slot.lower()]
                          for slot in slots if slot.lower() in self.entities}
        cache_key = self._prototype_cache_key(list(samples), entity_values)
        n = self._add_prototypes(label, expanded, self._prototype_k, cache_key)
        self.intents.add(label)
        self._store_context_gate(label, message)
        LOG.debug(f"Prototype store: added {n} prototype(s) for INTENT-4 template '{label}'")

    def _store_context_gate(self, label: str, message: Message) -> None:
        """Record OVOS-CONTEXT-1 §6/§6.1 ``requires_context`` /
        ``excludes_context`` declarations for ``label`` (if any) so they can be
        enforced at match time, plus the label's declared slot names so
        OVOS-CONTEXT-1 §7 can fill them from live context. Absent declarations
        clear any stale entry."""
        requires = message.data.get("requires_context")
        excludes = message.data.get("excludes_context")
        if requires or excludes:
            self._context_gates[label] = (requires or [], excludes or [])
        else:
            self._context_gates.pop(label, None)

        # Parse declared slot names from the ORIGINAL samples, before entity
        # expansion rewrites the ``{slot}`` placeholders into concrete values.
        slot_names: List[str] = []
        for sample in message.data.get("samples") or []:
            for slot in _SLOT_RE.findall(sample):
                if slot not in slot_names:
                    slot_names.append(slot)
        if slot_names:
            self._intent_slots[label] = slot_names
        else:
            self._intent_slots.pop(label, None)

    def _handle_intent4_register_entity(self, message: Message) -> None:
        """Register an entity value-set hint (§7). No-op in classifier mode."""
        topic = SpecMessage.ENTITY_REGISTER.value
        name: str = message.data.get("entity_name", "")
        samples = message.data.get("samples")
        if not name:
            self._intent4_warn(topic, message, "missing entity_name")
            return
        if not samples:  # missing or empty -> malformed (§7.2)
            self._intent4_warn(topic, message, "samples missing or empty")
            return
        if self.prototype_store is None:
            return  # classifier model is frozen; no slot-fill to perform
        values: set = set()
        for s in samples:
            if len(values) >= MAX_ENTITY_EXPANSIONS:
                break
            try:
                for v in islice(iter_expand(s), MAX_ENTITY_EXPANSIONS):
                    v = v.strip()
                    if v:
                        values.add(v)
            except Exception as exc:
                # skip the unparsable entry, keep the valid ones (§7.2)
                LOG.warning(
                    f"skipping malformed entity sample {s!r}: {exc} "
                    f"[skill_id={message.data.get('skill_id')!r} "
                    f"name={name!r} "
                    f"lang={message.data.get('lang')!r} topic={topic}]")
        if not values:  # zero valid entries -> malformed (§7.2)
            self._intent4_warn(topic, message, "no valid entity sample remains")
            return
        self.entities[name.lower()] = list(values)
        LOG.debug(f"Model2Vec: registered INTENT-4 entity '{name}' ({len(values)} values)")

    def _handle_intent4_deregister_intent(self, message: Message) -> None:
        """Remove one intent (§8.2)."""
        label = self._intent4_label(message)
        if not label:
            return
        if self.prototype_store is not None:
            self.prototype_store.remove(label)
            self._forget_pending(label)
        self.intents.discard(label)
        self._context_gates.pop(label, None)
        self._intent_slots.pop(label, None)
        self.excluded_keywords.pop(label, None)
        LOG.debug(f"Model2Vec: deregistered INTENT-4 intent '{label}'")

    def _handle_intent4_deregister_entity(self, message: Message) -> None:
        """Remove one entity (§8.3). No-op in classifier mode."""
        name: str = message.data.get("entity_name", "")
        if name:
            self.entities.pop(name.lower(), None)
            LOG.debug(f"Model2Vec: deregistered INTENT-4 entity '{name}'")

    def _handle_intent4_deregister_skill(self, message: Message) -> None:
        """Remove every intent and entity owned by a skill (§8.4)."""
        skill_id: str = message.data.get("skill_id") or message.context.get("skill_id", "")
        if not skill_id:
            return
        if self.prototype_store is not None:
            self.prototype_store.remove_skill(skill_id)
            self._forget_pending_skill(skill_id)
        self.intents = {i for i in self.intents if not i.startswith(skill_id + ":")}
        self._context_gates = {l: g for l, g in self._context_gates.items()
                               if not l.startswith(skill_id + ":")}
        self._intent_slots = {l: s for l, s in self._intent_slots.items()
                              if not l.startswith(skill_id + ":")}
        self.excluded_keywords = {i: kw for i, kw in self.excluded_keywords.items()
                                  if not i.startswith(skill_id + ":")}
        LOG.debug(f"Model2Vec: deregistered all INTENT-4 registrations for skill '{skill_id}'")

    def _handle_intent4_disable(self, message: Message) -> None:
        """Suppress an intent without losing its definition (§8.5).

        m2v keeps no separate enabled/disabled flag; suppression is realised
        by dropping the label from the match-eligible set (and its prototypes),
        mirroring deregistration. Re-enabling requires re-registration, which
        is how skills re-arm intents on this engine.
        """
        label = self._intent4_label(message)
        if not label:
            return
        if self.prototype_store is not None:
            self.prototype_store.remove(label)
            self._forget_pending(label)
        self.intents.discard(label)
        self._context_gates.pop(label, None)
        self._intent_slots.pop(label, None)
        self.excluded_keywords.pop(label, None)
        LOG.debug(f"Model2Vec: disabled INTENT-4 intent '{label}'")

    def _handle_intent4_enable(self, message: Message) -> None:
        """Re-arm a previously disabled intent (§8.5).

        In classifier mode the trained class is re-added to the eligible set.
        In prototype mode the prototypes were dropped on disable and can only
        be restored by re-registration, so this only restores label tracking.
        """
        label = self._intent4_label(message)
        if label and label not in self.ignore_labels:
            self.intents.add(label)
            LOG.debug(f"Model2Vec: enabled INTENT-4 intent '{label}'")

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _apply_special_label_map(self, label: str) -> Tuple[str, str]:
        """Return ``(skill_id, canonical_label)`` after applying ``self.label_map``.

        ``self.label_map`` entries are either the built-in ``(skill_id,
        canonical_label)`` tuples, or a ``skill_id:intent`` string (from a
        model manifest or user config). A string target that does not
        contain a colon does not name a `skill_id:intent` topic; it is used
        as-is (with a warning) rather than inventing one.
        """
        target = self.label_map.get(label)
        if target is None:
            return label.split(":")[0], label
        if isinstance(target, (list, tuple)) and len(target) == 2:
            skill_id, canonical_label = target
            return skill_id, canonical_label
        if isinstance(target, str):
            if ":" not in target:
                if label not in self._label_map_warned:
                    self._label_map_warned.add(label)
                    LOG.warning(f"Model2Vec: label_map target '{target}' for "
                                f"'{label}' is not in 'skill_id:intent' form; "
                                f"using as-is")
                return target, target
            return target.split(":")[0], target
        if label not in self._label_map_warned:
            self._label_map_warned.add(label)
            LOG.warning(f"Model2Vec: unsupported label_map target for "
                        f"'{label}': {target!r}; using as-is")
        return label.split(":")[0], label

    def _excluded_labels(self, utterance: str) -> List[str]:
        """Labels whose §6.1 ``blacklist`` phrases occur in *utterance*.

        Uses the same word-boundary convention as the padacioso engine's
        ``_filter``: single-word phrases must match a whole token, multi-word
        phrases match on a ``\\b``-delimited substring.
        """
        if not self.excluded_keywords:
            return []
        excluded: List[str] = []
        q_lower = utterance.lower()
        query_words = set(q_lower.split())
        for label, phrases in self.excluded_keywords.items():
            def _kw_hit(kw: str, _qw=query_words, _ql=q_lower) -> bool:
                kw = kw.lower()
                if " " not in kw:
                    return kw in _qw
                return bool(re.search(r"\b" + re.escape(kw) + r"\b", _ql))
            if any(_kw_hit(p) for p in phrases):
                excluded.append(label)
        return excluded

    def _session_blacklists(self, message: Optional[Message]) -> Tuple[frozenset, frozenset]:
        """``(blacklisted_intents, blacklisted_skills)`` for the caller's session.

        Mirrors adapt/padatious: candidates whose intent label or skill_id is
        blacklisted in the session are dropped at match time.
        """
        if message is None:
            return frozenset(), frozenset()
        try:
            sess = SessionManager.get(message)
        except Exception:
            return frozenset(), frozenset()
        return (frozenset(sess.blacklisted_intents or []),
                frozenset(sess.blacklisted_skills or []))

    def _match(self, utterance: str,
               message: Optional[Message] = None) -> Iterable[Tuple[str, str, float, Dict[str, Any]]]:
        """Yield ``(skill_id, label, score, slots)`` tuples sorted by score
        descending, where ``slots`` carries the OVOS-CONTEXT-1 §7
        context-supplied slot values for the label (empty when none apply).

        Candidates suppressed by a §6.1 template ``blacklist`` phrase, or by the
        session's ``blacklisted_intents`` / ``blacklisted_skills``, are dropped.

        Args:
            utterance: The utterance to match.
            message: The incoming bus message (used to read session.pipeline for
                     special-label gating, plus the session blacklists).
        """
        if not self._ensure_model():
            # cold start still warming up in the background (see
            # `_ensure_model`) -- no match for THIS utterance, matching
            # resumes automatically once the deferred load completes.
            return
        if self.prototype_store is not None:
            candidates = self._match_prototype(utterance, message)
        else:
            candidates = self._match_classifier(utterance, message)
        # OVOS-CONTEXT-1 §6/§6.1 context gate + OVOS-INTENT-4 §6.1 blacklist +
        # session blacklists — applied together so a candidate must survive all
        # three to be yielded. Surviving candidates additionally receive their
        # OVOS-CONTEXT-1 §7 context-supplied slots.
        #
        # The OVOS-INTENT-4 per-slot entity `.blacklist` remains a no-op for
        # m2v: this is a semantic *label* classifier and never extracts a slot
        # value from the utterance, so there is no utterance-supplied value to
        # exclude. Registered entities feed `_expand_entities` at registration
        # time to widen a label's prototypes; they are not extracted at match.
        #
        # §7 does apply: an m2v template may DECLARE a `{slot}`. Because the
        # utterance never fills that slot, live intent context is the only
        # source — so any declared slot with a live non-null context entry fills
        # directly (there is no utterance-produced value to override, §7). The
        # fill is independent of `requires_context`: the gate flags gate the
        # match, they do not scope the fill.
        intent_context = (SessionManager.get(message).intent_context or {}) \
            if (self._context_gates or self._intent_slots) else {}
        excluded = self._excluded_labels(utterance)
        blacklisted_intents, blacklisted_skills = self._session_blacklists(message)
        for skill_id, label, score in candidates:
            # `ignore_intents` (deny-list) applied post-mapping against the
            # canonical label so it works uniformly across classifier and
            # prototype mode, and covers special (OCP/stop/query) labels once
            # mapped. `valid_labels` is checked earlier, against the raw model
            # label, in `_match_classifier`/`_match_prototype` (see there for
            # why).
            if label in self.ignore_labels:
                LOG.debug(f"discarding match: {label} - in ignore_intents")
                continue
            gate = self._context_gates.get(label)
            if gate is not None:
                requires, excludes = gate
                if not gate_satisfied(intent_context, requires, excludes, owner_id=skill_id):
                    LOG.debug(f"discarding '{label}': CONTEXT-1 gate not satisfied")
                    continue
            if label in excluded:
                LOG.debug(f"discarding match: {label} - utterance hits §6.1 blacklist")
                continue
            if label in blacklisted_intents or skill_id in blacklisted_skills:
                LOG.debug(f"discarding match: {label} - blacklisted in session")
                continue
            slots: Dict[str, Any] = {}
            slot_names = self._intent_slots.get(label)
            if slot_names:
                slots = context_slot_candidates(intent_context, slot_names,
                                                owner_id=skill_id)
            yield skill_id, label, score, slots

    def _match_classifier(self, utterance: str,
                           message: Optional[Message] = None) -> Iterable[Tuple[str, str, float]]:
        """Classifier-mode matching using softmax probabilities."""
        inputs = [utterance]
        probs_ = self.model.predict_proba(inputs)
        # Include special-case labels gated by the session's active pipelines.
        special = self._allowed_special_labels(message)
        mask = np.isin(self.model.classes_, list(self.intents) + list(special))
        if not mask.any():
            LOG.warning("No model classes match registered intents")
            return
        classes = self.model.classes_[mask]
        probs = probs_[:, mask]
        # Renormalize probs over the surviving subset
        if self.config.get("renormalize"):
            row_sum = probs.sum(axis=1, keepdims=True)
            probs = np.where(row_sum > 0, probs / row_sum, probs)

        # Associate predictions with labels
        for input_text, prob_row in zip(inputs, probs):
            # Zip together class labels with their probabilities
            class_probs = list(zip(classes, prob_row))
            # Sort by probability descending
            class_probs.sort(key=lambda x: x[1], reverse=True)
            for label, prob in class_probs:
                LOG.debug(f"Match candidate: {label} - prob: {prob}")

                # `valid_labels` describes the model's raw label vocabulary
                # (labels.json), so it is checked BEFORE the special-label map
                # below rewrites e.g. "ocp:play" to "ovos.common_play.play_search"
                # — otherwise a manifest listing the raw training labels would
                # silently discard every OCP/stop/common_query match.
                if self.valid_labels is not None and label not in self.valid_labels:
                    LOG.debug(f"discarding match: {label} - not in valid_labels")
                    continue

                # HACK: special case for OCP, it isnt a regular intent
                skill_id, label = self._apply_special_label_map(label)

                yield skill_id, label, float(prob)

    def _match_prototype(self, utterance: str,
                         message: Optional[Message] = None) -> Iterable[Tuple[str, str, float]]:
        """Cosine nearest-neighbour against the prototype store.

        Special labels (ocp:play, common_query:common_query, stop:stop) are
        only forwarded when the matching downstream pipeline is present in the
        caller's session, consistent with classifier mode.
        """
        emb = self.model.encode([utterance], use_multiprocessing=False)[0]
        label_scores = self.prototype_store.scores(emb)
        special = self._allowed_special_labels(message)
        for label, score in sorted(label_scores.items(), key=lambda x: x[1], reverse=True):
            LOG.debug(f"Match candidate: {label} - cosine: {score:.4f}")
            if label in self.ignore_labels:
                continue
            # Gate special labels the same way as classifier mode
            if label in _SPECIAL_LABELS and label not in special:
                LOG.debug(f"discarding special label: {label} - not in session pipeline")
                continue
            # `valid_labels` is NOT applied here: it is a classifier-only
            # allow-list drawn from the model manifest's frozen training
            # vocabulary. Prototype labels are registered at runtime and are
            # legitimately absent from that manifest, so the prototype store
            # already IS the allow-list — every label it holds was explicitly
            # registered.
            skill_id, label = self._apply_special_label_map(label)
            yield skill_id, label, score

    # ------------------------------------------------------------------
    # Confidence-tier API
    # ------------------------------------------------------------------

    def match_high(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """
        Matches the most likely intent for a given list of utterances using Model2Vec.

        Args:
            utterances: A list of utterances to match against the model.
            lang: The language of the input utterance.
            message: The incoming message containing additional context.

        Returns:
            An IntentHandlerMatch if a high-confidence match is found, None otherwise.
        """
        if not utterances:
            return None
        min_conf = self.config.get("conf_high", 0.7)
        LOG.debug(f"Matching intents via Model2Vec (min_conf: {min_conf}) - {utterances[0]}")
        for skill_id, label, prob, slots in self._match(utterances[0], message):
            if prob < min_conf:
                LOG.debug(f"discarding match: {label} - confidence < {min_conf}")
                return None
            match = IntentHandlerMatch(
                match_type=label,
                match_data={"utterance": utterances[0], "confidence": prob, **slots},
                skill_id=skill_id or "ovos-m2v-pipeline",
                utterance=utterances[0]
            )
            LOG.debug(f"Match candidate: {match}")
            return match
        return None

    def match_medium(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """
        Matches the most likely intent for a given list of utterances using Model2Vec.

        Args:
            utterances: A list of utterances to match against the model.
            lang: The language of the input utterance.
            message: The incoming message containing additional context.

        Returns:
            An IntentHandlerMatch if a medium-confidence match is found, None otherwise.
        """
        if not utterances:
            return None
        min_conf = self.config.get("conf_medium", 0.5)
        LOG.debug(f"Matching intents via Model2Vec (min_conf: {min_conf}) - {utterances[0]}")
        for skill_id, label, prob, slots in self._match(utterances[0], message):
            if prob < min_conf:
                LOG.debug(f"discarding match: {label} - confidence < {min_conf}")
                return None
            match = IntentHandlerMatch(
                match_type=label,
                match_data={"utterance": utterances[0], "confidence": prob, **slots},
                skill_id=skill_id or "ovos-m2v-pipeline",
                utterance=utterances[0]
            )
            LOG.debug(f"Match: {match}")
            return match
        return None

    def match_low(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """
        Matches the most likely intent for a given list of utterances using Model2Vec.

        Args:
            utterances: A list of utterances to match against the model.
            lang: The language of the input utterance.
            message: The incoming message containing additional context.

        Returns:
            An IntentHandlerMatch if a low-confidence match is found, None otherwise.
        """
        if not utterances:
            return None
        min_conf = self.config.get("conf_low", 0.15)
        LOG.debug(f"Matching intents via Model2Vec (min_conf: {min_conf}) - {utterances[0]}")
        for skill_id, label, prob, slots in self._match(utterances[0], message):
            if prob < min_conf:
                LOG.debug(f"discarding match: {label} - confidence < {min_conf}")
                return None
            match = IntentHandlerMatch(
                match_type=label,
                match_data={"utterance": utterances[0], "confidence": prob, **slots},
                skill_id=skill_id or "ovos-m2v-pipeline",
                utterance=utterances[0]
            )
            LOG.debug(f"Match candidate: {match}")
            return match
        return None


class Model2VecPrototypePipeline(Model2VecIntentPipeline):
    """Prototype-mode Model2Vec pipeline exposed as a standalone OVOS plugin.

    Identical to ``Model2VecIntentPipeline`` with ``mode`` forced to
    ``"prototype"``.  Configuration is read from
    ``intents.ovos_m2v_prototype_pipeline`` so it can coexist with the
    classifier plugin in the same OVOS instance.

    Example ``mycroft.conf``::

        "intents": {
            "ovos-m2v-pipeline": {
                "model": "OpenVoiceOS/ovos-m2v-intents-multilingual"
            },
            "ovos-m2v-prototype-pipeline": {
                "model": "minishlab/M2V_multilingual_output",
                "prototype_k": 5
            }
        }
    """

    def __init__(
        self,
        bus: Optional[Union[MessageBusClient, FakeBus]] = None,
        config: Optional[Dict] = None,
    ) -> None:
        if config is None:
            config = (
                Configuration().get("intents", {}).get("ovos_m2v_prototype_pipeline") or {}
            )
        config["mode"] = "prototype"
        super().__init__(bus, config)
