"""m2v x nebulento fusion intent engine.

``Model2VecNebulentoFusionPipeline`` extends the prototype-mode m2v pipeline
(:class:`ovos_m2v_pipeline.Model2VecPrototypePipeline`) with a second,
independent signal -- fuzzy template matching via `nebulento
<https://github.com/OpenJarbas/nebulento>`_ -- fused at the *score* level
instead of the cascade fighters' pattern of trying one engine after another
and discarding whatever the first one produced. When the two signals agree
on a label they should reinforce each other; a cascade throws that agreement
away the moment the first engine returns a confident answer.

Algorithm
---------

1. **Registration.** Every registration path that already feeds the m2v
   prototype store (``padatious:register_intent``, the legacy Adapt/detach
   topics, and the OVOS-INTENT-4 ``ovos.intent.register.template`` /
   ``ovos.entity.register`` family) is reused unchanged -- this class only
   adds one extra step, in :meth:`_add_prototypes`: the exact same expanded,
   slot-bearing template strings that go into the m2v store are also handed
   to a single shared ``nebulento.IntentContainer`` via ``add_intent(label,
   samples)``. Nebulento does its own bracket/entity expansion internally,
   so passing already-expanded strings is a no-op for it. Deregistration
   (``detach_intent``, ``detach_skill``, the OVOS-INTENT-4 deregister/disable
   topics) mirrors the removal into the container the same way.

2. **Match.** The utterance is embedded once and scored against the m2v
   prototype store, producing ``s_m2v`` for every registered label. The top
   ``fusion_top_k`` labels by ``s_m2v`` (default 5, ``config["fusion_top_k"]``)
   are then looked up in the nebulento container's per-utterance fuzzy scores
   (one ``calc_intents()`` call scores every registered label at once) to get
   ``s_neb`` -- 0.0 when nebulento has no alignment for that specific label.
   Because the lookup is keyed by label, a candidate where nebulento's best
   alignment is for some OTHER label never contributes to this candidate: the
   match is label-constrained by construction, not by a shared top-1 pick.

   The fused confidence is a noisy-OR combination of the two signals when both
   exist for a label::

       conf = 1 - (1 - s_m2v) * (1 - s_neb)

   which is monotonically at least as high as either component (each term
   pulls the product of complements down towards 0, i.e. ``conf`` up towards
   1), so two independent signals agreeing on a label always outscore either
   one alone -- and degrades gracefully to the bare component when nebulento
   has no alignment for the label (``conf = s_m2v``). The winning candidate is
   the argmax over fused confidence across the top-K set.

3. **Slots.** Unlike the plain m2v engines (which never look at the
   utterance for entity values, only for embedding similarity), nebulento's
   fuzzy alignment extracts ``{slot}`` values from the utterance itself when
   the winning label's templates declare them. Those extracted values are
   attached to ``match_data`` alongside the OVOS-CONTEXT-1 §7 context-filled
   slots the base class already produces (context values win on conflict, on
   the theory that live conversational context is more authoritative than a
   fuzzy string alignment).

4. **Tiers.** Because a fused noisy-OR score sits structurally higher than
   either input signal on genuine agreement, the confidence tiers used here
   are correspondingly higher than the plain m2v defaults (0.7 / 0.5 / 0.15).
   Hand-worked cases with the formula above::

       s_m2v=0.55, s_neb=0.60 (agreement)  -> conf = 1-(0.45*0.40) = 0.82
       s_m2v=0.70, s_neb=0.70 (agreement)  -> conf = 1-(0.30*0.30) = 0.91
       s_m2v=0.50, s_neb=0.50 (agreement)  -> conf = 1-(0.50*0.50) = 0.75
       s_m2v=0.30, s_neb=0.30 (agreement)  -> conf = 1-(0.70*0.70) = 0.51
       s_m2v=0.45, s_neb=0.00 (m2v alone)  -> conf = 0.45

   ``conf_high`` defaults to 0.85: two individually-strong signals (>=0.7
   each) clear it, but a single strong m2v score with no nebulento alignment
   (0.9 alone) also still clears it on its own merit -- the tier does not
   *require* agreement, it merely rewards it. ``conf_medium`` defaults to
   0.65: two medium (~0.5) signals agreeing clear it, matching the intuition
   that "two plausible-but-not-certain signals pointing the same way" is a
   medium-confidence match. ``conf_low`` defaults to 0.4: two weak (~0.3)
   signals agreeing clear it (0.51) even though neither alone would, which is
   the whole point of fusing agreement instead of discarding it; a single
   weak m2v score with no nebulento echo (0.35) does not.

Optional dependency
--------------------

``nebulento`` is not a hard dependency of ``ovos-m2v-pipeline`` -- only this
one entry point needs it. It ships as the ``fusion`` extra
(``pip install ovos-m2v-pipeline[fusion]``) and is imported lazily, inside
``__init__``, so importing :mod:`ovos_m2v_pipeline` (and every other engine in
this package) works whether or not nebulento is installed. Constructing
:class:`Model2VecNebulentoFusionPipeline` without it raises ``ImportError``
with a message naming the extra to install.
"""
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_config.config import Configuration
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG

from ovos_m2v_pipeline import (
    Model2VecPrototypePipeline,
    _SPECIAL_LABELS,
)

#: Default per-utterance cap on how many m2v candidates get a nebulento
#: lookup. m2v's cosine ranking is cheap over the whole label set; nebulento's
#: fuzzy alignment is not, so only the labels m2v already considers plausible
#: are checked, not every registered label.
DEFAULT_FUSION_TOP_K = 5


class Model2VecNebulentoFusionPipeline(Model2VecPrototypePipeline):
    """Prototype-mode m2v fused with nebulento fuzzy template verification.

    See the module docstring for the full algorithm. Configuration is read
    from ``intents.ovos_m2v_nebulento_pipeline`` so it can coexist with the
    plain m2v classifier and prototype plugins in the same OVOS instance.

    Extra configuration keys (on top of everything
    :class:`~ovos_m2v_pipeline.Model2VecPrototypePipeline` accepts):

    ``fusion_top_k`` : int, default 5
        How many top m2v candidates get a nebulento lookup per utterance.
    ``conf_high`` / ``conf_medium`` / ``conf_low`` : float, defaults
        0.85 / 0.65 / 0.4 -- see the module docstring for the derivation.
    """

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        try:
            from nebulento import IntentContainer
        except ImportError as e:
            raise ImportError(
                "Model2VecNebulentoFusionPipeline requires the 'nebulento' "
                "package; install it via the 'fusion' extra: "
                "pip install ovos-m2v-pipeline[fusion]"
            ) from e
        if config is None:
            config = Configuration().get("intents", {}).get(
                "ovos_m2v_nebulento_pipeline") or {}
        #: shared fuzzy-matching container, one registered "intent" per m2v
        #: label; must exist before `super().__init__()` since registration
        #: handlers (wired inside it) call `_add_prototypes` immediately for
        #: skills that were already loaded.
        self.nebulento = IntentContainer()
        self.fusion_top_k: int = int(config.get("fusion_top_k", DEFAULT_FUSION_TOP_K))
        #: populated by `_match_prototype` (utterance-extracted entity values
        #: per winning label, from nebulento's alignment) and merged into
        #: `match_data` by the overridden `_match` below.
        self._fusion_entities: Dict[str, Dict[str, Any]] = {}
        super().__init__(bus, config)

    # ------------------------------------------------------------------
    # Registration mirroring
    # ------------------------------------------------------------------

    def _add_prototypes(self, label: str, sentences, k, cache_key) -> int:
        n = super()._add_prototypes(label, sentences, k, cache_key)
        if sentences:
            try:
                if label in self.nebulento.registered_intents:
                    self.nebulento.remove_intent(label)
                self.nebulento.add_intent(label, list(sentences))
            except Exception as exc:
                LOG.warning(f"nebulento fusion: failed to mirror templates "
                            f"for '{label}': {exc}")
        return n

    def _remove_nebulento_label(self, label: str) -> None:
        if label:
            self.nebulento.remove_intent(label)

    def _remove_nebulento_skill(self, skill_id: str) -> None:
        if not skill_id:
            return
        prefix = skill_id + ":"
        for label in list(self.nebulento.registered_intents):
            if label.startswith(prefix):
                self.nebulento.remove_intent(label)

    def _handle_detach_intent(self, message: Message) -> None:
        name: str = message.data.get("intent_name", "")
        if name.endswith(".intent"):
            name = name[:-len(".intent")]
        super()._handle_detach_intent(message)
        self._remove_nebulento_label(name)

    def _handle_detach_skill(self, message: Message) -> None:
        skill_id: str = message.data.get("skill_id") or message.context.get("skill_id", "")
        super()._handle_detach_skill(message)
        self._remove_nebulento_skill(skill_id)

    def _handle_intent4_deregister_intent(self, message: Message) -> None:
        label = self._intent4_label(message)
        super()._handle_intent4_deregister_intent(message)
        self._remove_nebulento_label(label)

    def _handle_intent4_deregister_skill(self, message: Message) -> None:
        skill_id: str = message.data.get("skill_id") or message.context.get("skill_id", "")
        super()._handle_intent4_deregister_skill(message)
        self._remove_nebulento_skill(skill_id)

    def _handle_intent4_disable(self, message: Message) -> None:
        label = self._intent4_label(message)
        super()._handle_intent4_disable(message)
        self._remove_nebulento_label(label)

    # ------------------------------------------------------------------
    # Confidence-tier defaults (see the module docstring for the derivation)
    # ------------------------------------------------------------------

    def match_high(self, utterances, lang, message):
        return self._match_tier("high", 0.85, utterances, lang, message)

    def match_medium(self, utterances, lang, message):
        return self._match_tier("medium", 0.65, utterances, lang, message)

    def match_low(self, utterances, lang, message):
        return self._match_tier("low", 0.4, utterances, lang, message)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def fuse(s_m2v: Optional[float], s_neb: Optional[float]) -> float:
        """Noisy-OR fusion of an m2v cosine score and a nebulento fuzzy score.

        ``conf = 1 - (1 - s_m2v) * (1 - s_neb)`` when both signals are given;
        degrades to whichever single signal is available otherwise. Always
        ``>=`` either given component (agreement is rewarded, never
        penalised) and stays within ``[0, 1]`` for inputs in ``[0, 1]``.
        """
        if s_m2v is None:
            return float(s_neb or 0.0)
        if s_neb is None:
            return float(s_m2v)
        return 1.0 - (1.0 - float(s_m2v)) * (1.0 - float(s_neb))

    def _match_prototype(self, utterance: str,
                          message: Optional[Message] = None
                          ) -> Iterable[Tuple[str, str, float]]:
        """Fuse m2v cosine similarity with nebulento fuzzy verification.

        Yields ``(skill_id, label, fused_conf)`` sorted by fused confidence
        descending. See the module docstring for the algorithm.
        """
        self._fusion_entities = {}
        emb = self.model.encode([utterance], use_multiprocessing=False)[0]
        label_scores = self.prototype_store.scores(emb)
        if not label_scores:
            return
        special = self._allowed_special_labels(message)
        top = sorted(label_scores.items(), key=lambda x: x[1],
                     reverse=True)[:self.fusion_top_k]

        neb_by_label: Dict[str, Dict[str, Any]] = {}
        if self.nebulento.registered_intents:
            for result in self.nebulento.calc_intents(utterance):
                neb_by_label[result["name"]] = result

        fused = []
        for label, s_m2v in top:
            neb = neb_by_label.get(label)
            s_neb = float(neb["conf"]) if neb is not None else None
            conf = self.fuse(float(s_m2v), s_neb)
            fused.append((label, conf, neb))
        fused.sort(key=lambda x: x[1], reverse=True)

        for label, conf, neb in fused:
            LOG.debug(f"Fusion match candidate: {label} - conf: {conf:.4f}")
            if label in self.ignore_labels:
                continue
            if label in _SPECIAL_LABELS and label not in special:
                LOG.debug(f"discarding special label: {label} - not in session pipeline")
                continue
            skill_id, canon = self._apply_special_label_map(label)
            if neb is not None and neb.get("entities"):
                self._fusion_entities[canon] = dict(neb["entities"])
            yield skill_id, canon, conf

    def _match(self, utterance: str,
               message: Optional[Message] = None
               ) -> Iterable[Tuple[str, str, float, Dict[str, Any]]]:
        """Wrap the base ``_match`` to merge nebulento-extracted slot values
        (populated by ``_match_prototype`` above) into each candidate's
        ``slots``, with the base class's OVOS-CONTEXT-1 §7 context-filled
        slots taking priority on key conflicts."""
        for skill_id, label, score, slots in super()._match(utterance, message):
            extra = self._fusion_entities.get(label)
            if extra:
                slots = {**extra, **slots}
            yield skill_id, label, score, slots
