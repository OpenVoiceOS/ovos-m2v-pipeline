"""Two-stage (hierarchical) prototype store for domain-routed intent matching.

Intents are grouped into *domains*. At query time a top-level router
selects the single best-matching domain, then intent resolution happens
only inside that domain's sub-store. This mirrors nebulento's standalone
:class:`~nebulento.HierarchicalIntentContainer`: the store owns its own
``domains`` dict of per-domain sub-stores plus a domain-fingerprint
store used as the router.

No training required. The static encoder produces the per-domain intent
embeddings, and the domain fingerprints are built from those same stored
anchors, so no sample is encoded twice.

The store has the mutation and scoring surface of
:class:`~ovos_m2v_pipeline.PrototypeIntentStore` (``add``, ``remove``,
``remove_skill`` and ``scores``, with the ``lang`` and ``cache_key``
keywords), so the prototype pipeline drives it without changes.
"""

import threading
from typing import Dict, List, Optional

import numpy as np

from ovos_m2v_pipeline import PrototypeIntentStore
from ovos_m2v_pipeline.cache import PrototypeCache
from ovos_m2v_pipeline.strategies import PrototypeStrategy, select_anchors


class HierarchicalPrototypeIntentStore:
    """Two-stage (hierarchical) prototype store grouped by domain.

    A label's domain is its ``skill_id``, the part before the first
    ``:``, unless :meth:`add` is given an explicit ``domain``.

    The router is a per-domain *fingerprint* store
    (:attr:`_domain_fingerprints`): the anchors of every intent in the
    domain, reduced by ``domain_strategy``, one entry per registration
    language. :meth:`calc_domain` picks the single highest-scoring domain.

    A ``domain_threshold`` gate rejects off-topic queries: when the best
    domain's fingerprint score is below the threshold, scoring returns
    no match. ``0.0`` (default) disables the gate.

    Example::

        from model2vec import StaticModel
        from ovos_m2v_pipeline import HierarchicalPrototypeIntentStore
        from ovos_m2v_pipeline.strategies import PrototypeStrategy

        model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
        store = HierarchicalPrototypeIntentStore(
            intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
            intent_tau=0.1,
            domain_threshold=0.2,
        )

        store.add(model, "media:play",     ["play {song}", "put on {song}"])
        store.add(model, "media:pause",    ["pause", "pause the music"])
        store.add(model, "home:lights_on", ["turn on the lights", "lights on"])

        scores = store.scores(model.encode(["play africa"])[0])
        # only the winning domain's intents are scored
    """

    _compose = staticmethod(PrototypeIntentStore._compose)
    _decompose = staticmethod(PrototypeIntentStore._decompose)

    def __init__(
        self,
        *,
        intent_strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        intent_top_k: int = 3,
        intent_tau: float = 0.1,
        domain_strategy: PrototypeStrategy = PrototypeStrategy.MEAN_CENTROID,
        domain_tau: float = 0.1,
        domain_threshold: float = 0.0,
        cache: Optional[PrototypeCache] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._intent_strategy = PrototypeStrategy(intent_strategy)
        self._intent_top_k = intent_top_k
        self._intent_tau = intent_tau
        self._domain_strategy = PrototypeStrategy(domain_strategy)
        #: Minimum fingerprint score the winning domain must reach for a
        #: query to be routed at all. Below it, scoring returns no match.
        self._domain_threshold = domain_threshold
        #: shared by every sub-store: labels are unique across domains, so
        #: one cache directory serves them all
        self.cache: Optional[PrototypeCache] = cache

        #: Per-domain intent stores, keyed by domain name.
        self.domains: Dict[str, PrototypeIntentStore] = {}
        #: bare label -> the domain that holds it
        self._domain_of_label: Dict[str, str] = {}
        #: Top-level router: one fingerprint entry per (domain, language).
        self._domain_fingerprints: PrototypeIntentStore = PrototypeIntentStore(
            strategy=self._domain_strategy,
            top_k=intent_top_k,
            tau=domain_tau,
        )

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def intent_strategy(self) -> PrototypeStrategy:
        return self._intent_strategy

    @property
    def domain_threshold(self) -> float:
        return self._domain_threshold

    def __len__(self) -> int:
        """Total number of prototypes across every domain's sub-store."""
        return sum(len(s) for s in list(self.domains.values()))

    @property
    def unique_labels(self) -> np.ndarray:
        """All intent labels across every domain (sorted, deduplicated)."""
        labels: set = set()
        for store in list(self.domains.values()):
            labels.update(str(lbl) for lbl in store.unique_labels)
        return np.asarray(sorted(labels), dtype=object)

    @staticmethod
    def domain_of(label: str) -> str:
        """The default domain of *label*: its ``skill_id`` prefix."""
        return label.split(":", 1)[0]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _sub_store(self, label: str, domain: Optional[str]) -> PrototypeIntentStore:
        """The sub-store for *label*, created on first use. A label that is
        registered again under another domain leaves the old domain."""
        domain = domain or self.domain_of(label)
        with self._lock:
            previous = self._domain_of_label.get(label)
            if previous is not None and previous != domain:
                self._remove_from(previous, label)
            if domain not in self.domains:
                self.domains[domain] = PrototypeIntentStore(
                    strategy=self._intent_strategy,
                    top_k=self._intent_top_k,
                    tau=self._intent_tau,
                    cache=self.cache,
                )
            self._domain_of_label[label] = domain
            return self.domains[domain]

    def add(self, model, label: str, sentences: List[str],
            k: Optional[int] = None, random_state: int = 42, *,
            cache_key: Optional[str] = None, lang: Optional[str] = None,
            domain: Optional[str] = None) -> int:
        """Embed *sentences* and add or replace the prototypes for *label*.

        The contract is that of :meth:`PrototypeIntentStore.add`. ``domain``
        overrides the domain taken from the label's ``skill_id`` prefix.
        Returns the number of prototypes added.
        """
        if not sentences:
            return 0
        sub = self._sub_store(label, domain)
        n = sub.add(model, label, sentences, k=k, random_state=random_state,
                    cache_key=cache_key, lang=lang)
        self._refresh_fingerprint(self._domain_of_label[label])
        return n

    def _add_anchors(self, internal_label: str, anchors: np.ndarray) -> int:
        """Insert precomputed anchors (the rows of a prebuilt artifact) for
        an internal label key, as :meth:`PrototypeIntentStore._add_anchors`."""
        label, _ = self._decompose(internal_label)
        sub = self._sub_store(label, None)
        n = sub._add_anchors(internal_label, anchors)
        self._refresh_fingerprint(self._domain_of_label[label])
        return n

    def _remove_from(self, domain: str, label: str) -> None:
        sub = self.domains.get(domain)
        if sub is None:
            return
        sub.remove(label)
        if not len(sub):
            self.domains.pop(domain, None)
        self._refresh_fingerprint(domain)

    def remove(self, label: str) -> None:
        """Remove all prototypes for *label*, in every language."""
        with self._lock:
            domain = self._domain_of_label.pop(label, None) or self.domain_of(label)
            self._remove_from(domain, label)

    def remove_skill(self, skill_id: str) -> None:
        """Remove every label prefixed ``<skill_id>:``, from any domain."""
        prefix = skill_id + ":"
        with self._lock:
            self._domain_of_label = {lbl: d for lbl, d in self._domain_of_label.items()
                                     if not lbl.startswith(prefix)}
            for domain in list(self.domains):
                sub = self.domains[domain]
                before = len(sub)
                sub.remove_skill(skill_id)
                if len(sub) == before:
                    continue
                if not len(sub):
                    self.domains.pop(domain)
                self._refresh_fingerprint(domain)

    def remove_domain(self, domain: str) -> None:
        """Remove a domain and all its intents."""
        with self._lock:
            sub = self.domains.pop(domain, None)
            self._domain_of_label = {lbl: d for lbl, d in self._domain_of_label.items()
                                     if d != domain}
            if sub is not None and self.cache is not None:
                for label in sub.unique_labels:
                    self.cache.remove(str(label))
            self._domain_fingerprints.remove(domain)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calc_domain(self, query_embedding: np.ndarray,
                    lang: Optional[str] = None) -> Optional[str]:
        """Return the single best-matching domain name for a query.

        Returns ``None`` when no domain is registered, or when the best
        fingerprint score is below :attr:`domain_threshold`.
        """
        fp_scores = self._domain_fingerprints.scores(query_embedding, lang=lang)
        if not fp_scores:
            return None
        domain, score = max(fp_scores.items(), key=lambda kv: (kv[1], kv[0]))
        if score < self._domain_threshold:
            return None
        return domain

    def scores(self, query_embedding: np.ndarray, lang: Optional[str] = None,
               domain: Optional[str] = None) -> Dict[str, float]:
        """Return ``{intent_label: score}`` for the routed domain only.

        Args:
            query_embedding: Raw (unnormalised) query embedding vector.
            lang: The query's BCP-47 tag. Both stages apply it as
                :meth:`PrototypeIntentStore.scores` does.
            domain: If given, skip the top-level router and score only
                this domain. This also skips the ``domain_threshold`` gate.
        """
        routed = domain if domain is not None else self.calc_domain(query_embedding, lang)
        sub = self.domains.get(routed) if routed is not None else None
        if sub is None:
            return {}
        return sub.scores(query_embedding, lang=lang)

    def calc_intent(self, query_embedding: np.ndarray, lang: Optional[str] = None,
                    domain: Optional[str] = None) -> Optional[str]:
        """Argmax over :meth:`scores`. Returns ``None`` if nothing scores."""
        scored = self.scores(query_embedding, lang=lang, domain=domain)
        if not scored:
            return None
        return max(scored, key=scored.get)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _refresh_fingerprint(self, domain: str) -> None:
        """Rebuild the router entries of *domain* from the anchors its
        sub-store holds, one entry per registration language."""
        with self._lock:
            self._domain_fingerprints.remove(domain)
            sub = self.domains.get(domain)
            if sub is None:
                return
            with sub._lock:
                embeddings = sub.embeddings
                langs = sub._entry_langs.tolist()
            for lang in set(langs):
                rows = embeddings[np.array([entry == lang for entry in langs])]
                anchors = select_anchors(rows, self._domain_strategy).astype(np.float32)
                self._domain_fingerprints._add_anchors(self._compose(domain, lang), anchors)
