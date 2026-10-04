"""Two-stage (hierarchical) trained intent classifier.

Mirrors :class:`~ovos_m2v_pipeline.hierarchical_store.HierarchicalPrototypeIntentStore`
but the routing and intent layers are supervised classifiers (scikit-learn
``LogisticRegression``) trained on top of a model2vec static encoder, rather
than centroid/anchor cosine matching.

Layout
------
A trained bundle on disk looks like::

    <bundle>/
        manifest.json              # version + domain list + threshold
        domain/                    # domain classifier (joblib)
            classifier.joblib
        intent/
            <domain_a>/
                classifier.joblib
            <domain_b>/
                classifier.joblib

Each ``classifier.joblib`` is a single scikit-learn ``LogisticRegression``
fitted on raw embeddings (no pipeline wrapper). At inference time the caller
supplies the embedding directly — the encoder is shared with the pipeline
that owns this classifier.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class HierarchicalIntentClassifier:
    """Supervised two-stage classifier: domain router + per-domain intent heads.

    At inference time the domain classifier scores the query embedding and
    picks a single best domain. If its softmax score is below
    ``domain_threshold`` the query is rejected (returned as no-match). Otherwise
    the matching per-domain intent classifier resolves the intent within that
    domain.
    """

    def __init__(
        self,
        domain_threshold: float = 0.0,
        domain_classifier=None,
        intent_classifiers: Optional[Dict[str, object]] = None,
    ) -> None:
        self.domain_threshold: float = float(domain_threshold)
        self.domain_classifier = domain_classifier
        self.intent_classifiers: Dict[str, object] = dict(intent_classifiers or {})

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def domains(self) -> List[str]:
        return sorted(self.intent_classifiers.keys())

    @property
    def unique_labels(self) -> np.ndarray:
        labels: set = set()
        for clf in self.intent_classifiers.values():
            classes = getattr(clf, "classes_", [])
            labels.update(str(c) for c in classes)
        return np.asarray(sorted(labels), dtype=object)

    def __len__(self) -> int:
        return sum(len(getattr(c, "classes_", [])) for c in self.intent_classifiers.values())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _split_label(label: str) -> Tuple[str, str]:
        """Split ``<skill_id>:<intent>`` (else ``<domain>.<intent>``) into its
        domain and the full label; a label with neither separator is its
        own domain. A skill id holds dots, so ``:`` is tried first."""
        for sep in (":", "."):
            if sep in label:
                d, _ = label.split(sep, 1)
                return d, label
        return label, label

    @classmethod
    def train(
        cls,
        embeddings: np.ndarray,
        labels: List[str],
        *,
        domain_threshold: float = 0.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> "HierarchicalIntentClassifier":
        """Fit a domain classifier + per-domain intent classifiers.

        Args:
            embeddings: ``(n_samples, dim)`` numpy array of model2vec embeddings.
            labels: parallel list of ``<domain>.<intent>`` labels.
            domain_threshold: softmax cutoff for the domain router.
            max_iter: LR ``max_iter``.
            random_state: LR ``random_state``.
        """
        from sklearn.linear_model import LogisticRegression

        embeddings = np.asarray(embeddings)
        labels = list(labels)
        if len(embeddings) != len(labels):
            raise ValueError("embeddings and labels must have the same length")

        domains = [cls._split_label(l)[0] for l in labels]

        domain_clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
        domain_clf.fit(embeddings, domains)

        intent_clfs: Dict[str, object] = {}
        unique_domains = sorted(set(domains))
        for d in unique_domains:
            mask = np.asarray([dd == d for dd in domains])
            X_d = embeddings[mask]
            y_d = [labels[i] for i, m in enumerate(mask) if m]
            unique_intents = sorted(set(y_d))
            if len(unique_intents) == 1:
                # LogisticRegression refuses 1-class problems — wrap in a
                # tiny constant predictor with the same API surface we need.
                intent_clfs[d] = _ConstantClassifier(unique_intents[0])
            else:
                clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
                clf.fit(X_d, y_d)
                intent_clfs[d] = clf

        return cls(
            domain_threshold=domain_threshold,
            domain_classifier=domain_clf,
            intent_classifiers=intent_clfs,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calc_domain(self, embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Return ``(domain, score)`` or ``None`` if below threshold."""
        if self.domain_classifier is None or not self.intent_classifiers:
            return None
        x = np.atleast_2d(embedding)
        probs = self.domain_classifier.predict_proba(x)[0]
        classes = list(self.domain_classifier.classes_)
        idx = int(np.argmax(probs))
        score = float(probs[idx])
        if score < self.domain_threshold:
            return None
        return classes[idx], score

    def predict(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Two-stage predict: ``(intent_label, confidence)`` or ``(None, 0.0)``."""
        routed = self.calc_domain(embedding)
        if routed is None:
            return None, 0.0
        domain, dscore = routed
        clf = self.intent_classifiers.get(domain)
        if clf is None:
            return None, 0.0
        x = np.atleast_2d(embedding)
        probs = clf.predict_proba(x)[0]
        classes = list(clf.classes_)
        idx = int(np.argmax(probs))
        # Multiply domain confidence by intent confidence to produce a
        # single end-to-end score in [0, 1].
        return str(classes[idx]), float(probs[idx]) * dscore

    def predict_proba(self, embedding: np.ndarray) -> Dict[str, float]:
        """Return ``{intent_label: score}`` for the routed domain only."""
        routed = self.calc_domain(embedding)
        if routed is None:
            return {}
        domain, dscore = routed
        clf = self.intent_classifiers.get(domain)
        if clf is None:
            return {}
        x = np.atleast_2d(embedding)
        probs = clf.predict_proba(x)[0]
        classes = list(clf.classes_)
        return {str(c): float(p) * dscore for c, p in zip(classes, probs)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the bundle to ``path/``. Layout described in the module docstring."""
        import joblib

        os.makedirs(path, exist_ok=True)
        domain_dir = os.path.join(path, "domain")
        intent_dir = os.path.join(path, "intent")
        os.makedirs(domain_dir, exist_ok=True)
        os.makedirs(intent_dir, exist_ok=True)

        if self.domain_classifier is not None:
            joblib.dump(self.domain_classifier, os.path.join(domain_dir, "classifier.joblib"))

        for domain, clf in self.intent_classifiers.items():
            d_dir = os.path.join(intent_dir, domain)
            os.makedirs(d_dir, exist_ok=True)
            joblib.dump(clf, os.path.join(d_dir, "classifier.joblib"))

        manifest = {
            "format": "ovos-m2v-hierarchical-intent/1",
            "domain_threshold": self.domain_threshold,
            "domains": sorted(self.intent_classifiers.keys()),
        }
        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "HierarchicalIntentClassifier":
        """Load a bundle previously written by :meth:`save`."""
        import joblib

        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)

        domain_path = os.path.join(path, "domain", "classifier.joblib")
        domain_clf = joblib.load(domain_path) if os.path.isfile(domain_path) else None

        intent_clfs: Dict[str, object] = {}
        intent_root = os.path.join(path, "intent")
        if os.path.isdir(intent_root):
            for d in sorted(os.listdir(intent_root)):
                clf_path = os.path.join(intent_root, d, "classifier.joblib")
                if os.path.isfile(clf_path):
                    intent_clfs[d] = joblib.load(clf_path)

        return cls(
            domain_threshold=float(manifest.get("domain_threshold", 0.0)),
            domain_classifier=domain_clf,
            intent_classifiers=intent_clfs,
        )


class _ConstantClassifier:
    """Tiny stand-in for single-class domains (LR refuses 1-class problems)."""

    def __init__(self, label: str) -> None:
        self.classes_ = np.asarray([label], dtype=object)
        self._label = label

    def predict(self, X):  # noqa: N803
        n = np.atleast_2d(X).shape[0]
        return np.asarray([self._label] * n, dtype=object)

    def predict_proba(self, X):  # noqa: N803
        n = np.atleast_2d(X).shape[0]
        return np.ones((n, 1), dtype=np.float32)
