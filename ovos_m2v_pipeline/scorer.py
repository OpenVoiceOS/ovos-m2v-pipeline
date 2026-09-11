"""Standalone, bus-free prototype scoring for external evaluators.

``Model2VecPrototypePipeline`` only exposes its matching logic through
OVOS bus events (``padatious:register_intent`` / ``ovos.intent.register.*``
in, ``complete_intent_failure`` / skill events out). A harness that wants
to *measure* the shipping prototype-mode algorithm -- ovoscope's
multi-engine runner, the intent-matching arena -- has no way to drive it
without standing up a full bus and a skill, and has historically
reimplemented "prototype scoring" from scratch (typically centroid
cosine). That reimplementation is a different algorithm from what
``PrototypeIntentStore`` actually does (per-sample max-over-all cosine
with bounded template/entity expansion, not a mean centroid), so any
numbers it produces describe the reimplementation, not the pipeline.

``PrototypeScorer`` wraps a bus-free ``Model2VecIntentPipeline`` (built
with ``bus=None``, which the base class turns into an in-memory
``FakeBus``) and drives its *real* OVOS-INTENT-4 registration handler
(``_handle_intent4_register_template``) and the real
``PrototypeIntentStore.scores()`` directly, via synthetic
``Message`` objects -- the exact same code path a live skill's bus
registration triggers, so external scoring and live dispatch can never
drift apart.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ovos_bus_client.message import Message
from ovos_spec_tools import SpecMessage

from ovos_m2v_pipeline import DEFAULT_MULTILINGUAL, Model2VecIntentPipeline
from ovos_m2v_pipeline.strategies import PrototypeStrategy

#: match() tier name -> the pipeline config key match_high/medium/low read
#: their threshold from.
_TIER_CONF_KEYS: Dict[str, str] = {
    "high": "conf_high",
    "medium": "conf_medium",
    "low": "conf_low",
}
_TIER_DEFAULTS: Dict[str, float] = {
    "high": 0.7,
    "medium": 0.5,
    "low": 0.15,
}


class PrototypeScorer:
    """Score utterances against zero-shot prototype intents, no bus required.

    ``add_intent`` registers example utterances exactly the way a live
    ``padatious:register_intent`` / OVOS-INTENT-4 template registration
    would (bracket expansion via ``iter_expand``, entity slot-fill via
    the pipeline's ``_expand_entities``, bounded by
    ``MAX_ENTITY_EXPANSIONS``), and ``score`` / ``match`` read the same
    ``PrototypeIntentStore.scores()`` a live match uses.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        lang: Optional[str] = None,
        *,
        conf_high: float = _TIER_DEFAULTS["high"],
        conf_medium: float = _TIER_DEFAULTS["medium"],
        conf_low: float = _TIER_DEFAULTS["low"],
        prototype_strategy: str = PrototypeStrategy.MAX_OVER_ALL.value,
        prototype_k: Optional[int] = None,
        prototype_top_k: int = 3,
        prototype_tau: float = 0.1,
        prototype_cache: bool = False,
    ) -> None:
        #: default lang threaded through ``add_intent``/``score``/``match``
        #: when the caller omits one. Not yet used to partition the store
        #: (the store has no per-language partitioning on `dev` as of this
        #: writing); kept as an explicit parameter now so that landing
        #: per-language partitioning is a signature-compatible change here.
        self.lang = lang
        config = {
            "mode": "prototype",
            "model": model or DEFAULT_MULTILINGUAL,
            "conf_high": conf_high,
            "conf_medium": conf_medium,
            "conf_low": conf_low,
            "prototype_strategy": prototype_strategy,
            "prototype_k": prototype_k,
            "prototype_top_k": prototype_top_k,
            "prototype_tau": prototype_tau,
            "prototype_cache": prototype_cache,
        }
        # bus=None -> ConfidenceMatcherPipeline.__init__ builds an in-memory
        # FakeBus; the registration handlers below are called directly
        # rather than dispatched, so nothing ever touches a real bus.
        self._pipeline = Model2VecIntentPipeline(bus=None, config=config)
        # Load eagerly (rather than the pipeline's own on-first-utterance
        # deferred load): an evaluator wants a ready-to-score object and a
        # predictable failure point, not a silently-skipped first score().
        if not self._pipeline._ensure_model(background_ok=False):
            raise RuntimeError(
                f"failed to load Model2Vec model '{config['model']}'; see "
                f"the logged exception for the underlying cause"
            )

    @staticmethod
    def _split_label(label: str) -> Tuple[str, str]:
        """Split *label* into the ``(skill_id, intent_name)`` pair the
        OVOS-INTENT-4 registration handler expects.

        ``"skill_id:intent_name"`` is split on the first colon, matching
        the canonical bus label form. A bare label with no colon gets a
        fixed ``"external"`` skill_id so the resulting canonical label
        (``f"{skill_id}:{intent_name}"``) is still unambiguous.
        """
        if ":" in label:
            skill_id, intent_name = label.split(":", 1)
            return skill_id, intent_name
        return "external", label

    def add_intent(
        self,
        label: str,
        samples: List[str],
        lang: Optional[str] = None,
        entities: Optional[Dict[str, List[str]]] = None,
    ) -> int:
        """Register *samples* as prototypes for *label*.

        Runs the real OVOS-INTENT-4 template registration handler
        (``_handle_intent4_register_template``): bracket-expansion via
        ``iter_expand``, ``{slot}`` entity substitution via
        ``_expand_entities``, and ingestion into ``PrototypeIntentStore``
        via ``_add_prototypes`` -- bounded by ``MAX_ENTITY_EXPANSIONS`` at
        every step, identically to a live registration.

        *entities* (optional ``{entity_name: [values, ...]}``) is
        registered first via the matching OVOS-INTENT-4 entity handler so
        ``{slot}`` placeholders in *samples* resolve.

        Returns the number of prototypes stored for the resulting
        canonical ``skill_id:intent_name`` label.
        """
        pipeline = self._pipeline
        use_lang = lang or self.lang or "en-us"
        if entities:
            for name, values in entities.items():
                pipeline._handle_intent4_register_entity(Message(
                    SpecMessage.ENTITY_REGISTER.value,
                    data={"entity_name": name, "samples": list(values),
                          "lang": use_lang},
                ))
        skill_id, intent_name = self._split_label(label)
        pipeline._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={"skill_id": skill_id, "intent_name": intent_name,
                  "samples": list(samples), "lang": use_lang},
        ))
        canonical = f"{skill_id}:{intent_name}"
        store = pipeline.prototype_store
        return int((store.labels == canonical).sum()) if len(store) else 0

    def score(self, utterance: str, lang: Optional[str] = None) -> List[Tuple[str, float]]:
        """Return ``[(label, cosine), ...]`` ranked highest-first.

        Encodes *utterance* and calls the real
        ``PrototypeIntentStore.scores()`` -- the same per-label
        max-cosine-over-anchors lookup ``_match_prototype`` uses on the
        bus dispatch path, minus the OCP/common-query/stop special-label
        session gating and the ``label_map`` bus-topic remap (both are
        bus-dispatch concerns; a standalone evaluator wants the raw
        registered labels back).
        """
        pipeline = self._pipeline
        if not pipeline._ensure_model(background_ok=False):
            raise RuntimeError("Model2Vec model is not loaded")
        emb = pipeline.model.encode([utterance], use_multiprocessing=False)[0]
        scores = pipeline.prototype_store.scores(emb)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def match(
        self,
        utterance: str,
        lang: Optional[str] = None,
        tier: str = "medium",
    ) -> Optional[str]:
        """Return the top label if it clears *tier*'s confidence threshold, else ``None``.

        Mirrors ``match_high``/``match_medium``/``match_low``: only the
        single highest-scoring label is ever considered (a below-threshold
        top score is a miss, not a reason to fall through to the runner-up),
        and the threshold comes from the same ``conf_high``/``conf_medium``/
        ``conf_low`` config keys those methods read.
        """
        conf_key = _TIER_CONF_KEYS.get(tier)
        if conf_key is None:
            raise ValueError(
                f"unknown tier {tier!r}; expected one of {sorted(_TIER_CONF_KEYS)}"
            )
        min_conf = self._pipeline.config.get(conf_key, _TIER_DEFAULTS[tier])
        ranked = self.score(utterance, lang=lang)
        if not ranked:
            return None
        label, conf = ranked[0]
        if conf < min_conf:
            return None
        return label
