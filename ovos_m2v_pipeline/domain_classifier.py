"""Domain (parallel-argmax) trained intent classifier.

A flat-API variant of :class:`~ovos_m2v_pipeline.hierarchical_classifier.HierarchicalIntentClassifier`
that drops the top-level domain router. Every per-domain intent classifier
is trained on the subset of samples whose label belongs to that domain,
then at inference time **every** per-domain classifier scores the query
and a single global argmax over their softmax outputs picks the winner.

Why this exists
---------------
Per-domain training fits each classifier on a different sample set, so its
decision boundary genuinely differs from a global flat classifier. The
parallel-argmax variant keeps that per-domain fitting benefit but removes
the two-stage routing step — useful when the top-level domain router is
the weakest link in the hierarchical setup, and when adding a new skill
should only require retraining one per-domain classifier.

Layout
------
A trained bundle on disk looks like::

    <bundle>/
        manifest.json              # version + domain list
        intent/
            <domain_a>/classifier.joblib
            <domain_b>/classifier.joblib
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ovos_m2v_pipeline.hierarchical_classifier import _ConstantClassifier


class DomainIntentClassifier:
    """Parallel per-domain trained classifier — global argmax across all heads.

    Construction parameters mirror :class:`HierarchicalIntentClassifier` minus
    the top-level domain classifier.
    """

    def __init__(
        self,
        intent_classifiers: Optional[Dict[str, object]] = None,
    ) -> None:
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
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> "DomainIntentClassifier":
        """Fit one classifier per domain (no top-level router)."""
        from sklearn.linear_model import LogisticRegression

        embeddings = np.asarray(embeddings)
        labels = list(labels)
        if len(embeddings) != len(labels):
            raise ValueError("embeddings and labels must have the same length")

        domains = [cls._split_label(l)[0] for l in labels]
        intent_clfs: Dict[str, object] = {}
        for d in sorted(set(domains)):
            mask = np.asarray([dd == d for dd in domains])
            X_d = embeddings[mask]
            y_d = [labels[i] for i, m in enumerate(mask) if m]
            unique_intents = sorted(set(y_d))
            if len(unique_intents) == 1:
                intent_clfs[d] = _ConstantClassifier(unique_intents[0])
            else:
                clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
                clf.fit(X_d, y_d)
                intent_clfs[d] = clf

        return cls(intent_classifiers=intent_clfs)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Run every per-domain classifier; return the global argmax label."""
        if not self.intent_classifiers:
            return None, 0.0
        scores = self.predict_proba(embedding)
        if not scores:
            return None, 0.0
        label = max(scores, key=scores.get)
        return label, float(scores[label])

    def predict_proba(self, embedding: np.ndarray) -> Dict[str, float]:
        """Return ``{intent_label: score}`` aggregated across every domain head."""
        if not self.intent_classifiers:
            return {}
        x = np.atleast_2d(embedding)
        out: Dict[str, float] = {}
        for clf in self.intent_classifiers.values():
            probs = clf.predict_proba(x)[0]
            classes = list(clf.classes_)
            for c, p in zip(classes, probs):
                out[str(c)] = float(p)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the bundle to ``path/``. Layout described in the module docstring."""
        import joblib

        os.makedirs(path, exist_ok=True)
        intent_dir = os.path.join(path, "intent")
        os.makedirs(intent_dir, exist_ok=True)

        for domain, clf in self.intent_classifiers.items():
            d_dir = os.path.join(intent_dir, domain)
            os.makedirs(d_dir, exist_ok=True)
            joblib.dump(clf, os.path.join(d_dir, "classifier.joblib"))

        manifest = {
            "format": "ovos-m2v-domain-intent/1",
            "domains": sorted(self.intent_classifiers.keys()),
        }
        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "DomainIntentClassifier":
        """Load a bundle previously written by :meth:`save`."""
        import joblib

        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fh:
            json.load(fh)

        intent_clfs: Dict[str, object] = {}
        intent_root = os.path.join(path, "intent")
        if os.path.isdir(intent_root):
            for d in sorted(os.listdir(intent_root)):
                clf_path = os.path.join(intent_root, d, "classifier.joblib")
                if os.path.isfile(clf_path):
                    intent_clfs[d] = joblib.load(clf_path)

        return cls(intent_classifiers=intent_clfs)
