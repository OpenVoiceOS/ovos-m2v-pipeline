"""Unit tests for OVOS-INTENT-4 template/entity registration adoption.

m2v is a TEMPLATE-style engine (it matches on example utterances), so it
consumes `ovos.intent.register.template` (§6) and `ovos.entity.register`
(§7) in addition to the legacy `padatious:register_intent` topics. It does
NOT consume `ovos.intent.register.keyword` (§11).
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_spec_tools import SpecMessage


def _make_prototype_pipeline(config=None):
    """Prototype-mode pipeline with a mocked StaticModel + FakeBus."""
    config = config or {}
    config.setdefault("model", "fake-embed-model")
    config["mode"] = "prototype"

    mock_embed_model = MagicMock()
    # identity rows so each sample becomes a distinct unit prototype
    mock_embed_model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)

    fake_m2v = MagicMock()
    fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch.dict(sys.modules, {"model2vec": fake_m2v}):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)
    pipeline.model = mock_embed_model
    return pipeline


def _make_classifier_pipeline(config=None):
    """Classifier-mode pipeline with a mocked StaticModelPipeline + FakeBus."""
    config = config or {}
    config.setdefault("model", "fake-model")

    mock_model = MagicMock()
    mock_model.classes_ = np.array([])
    mock_model.predict_proba.return_value = np.array([[]])

    with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
         patch("ovos_m2v_pipeline.Configuration", return_value={}):
        MockSMP.from_pretrained.return_value = mock_model
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)
    pipeline.model = mock_model
    return pipeline


class TestIntent4Subscriptions(unittest.TestCase):
    """The INTENT-4 topics are subscribed (and the keyword topic is NOT)."""

    def _registered_topics(self, pipeline):
        # FakeBus stores handlers in .ee (EventEmitter); fall back to events dict
        try:
            return set(pipeline.bus.ee._events.keys())
        except AttributeError:
            return set(getattr(pipeline.bus, "events", {}).keys())

    def test_prototype_mode_subscribes_template_not_keyword(self):
        p = _make_prototype_pipeline()
        topics = self._registered_topics(p)
        self.assertIn(SpecMessage.INTENT_REGISTER_TEMPLATE.value, topics)
        self.assertIn(SpecMessage.ENTITY_REGISTER.value, topics)
        self.assertIn(SpecMessage.INTENT_DEREGISTER.value, topics)
        self.assertIn(SpecMessage.SKILL_DEREGISTER.value, topics)
        self.assertNotIn(SpecMessage.INTENT_REGISTER_KEYWORD.value, topics)

    def test_classifier_mode_subscribes_template_not_keyword(self):
        p = _make_classifier_pipeline()
        topics = self._registered_topics(p)
        self.assertIn(SpecMessage.INTENT_REGISTER_TEMPLATE.value, topics)
        self.assertNotIn(SpecMessage.INTENT_REGISTER_KEYWORD.value, topics)

    def test_legacy_topics_still_subscribed(self):
        p = _make_prototype_pipeline()
        topics = self._registered_topics(p)
        self.assertIn("padatious:register_intent", topics)
        self.assertIn("detach_intent", topics)
        self.assertIn("detach_skill", topics)


class TestIntent4TemplateRegistration(unittest.TestCase):
    def _register(self, p, skill_id="music.skill", intent_name="play_music",
                  samples=None, lang="en-US"):
        msg = Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={
                "skill_id": skill_id,
                "intent_name": intent_name,
                "lang": lang,
                "samples": samples if samples is not None else ["play music", "put on a song"],
            },
            context={"skill_id": skill_id},
        )
        p._handle_intent4_register_template(msg)

    def test_register_accepted_and_label_tracked(self):
        p = _make_prototype_pipeline()
        self._register(p)
        self.assertIn("music.skill:play_music", p.intents)
        self.assertIn("music.skill:play_music", p.prototype_store.unique_labels)

    def test_registered_intent_matches_utterance(self):
        """Register via the template topic, then match an utterance.

        The mock encoder maps the Nth registered sample to basis vector e_N.
        With max_over_all the query that equals one prototype yields cosine 1.0.
        """
        p = _make_prototype_pipeline()
        # encode returns deterministic basis vectors; register a single sample
        self._register(p, samples=["play music"])
        # query embedding identical to the stored prototype -> cosine 1.0
        p.model.encode.side_effect = None
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("play music"))
        self.assertEqual(len(results), 1)
        skill_id, label, score, slots = results[0]
        self.assertEqual(label, "music.skill:play_music")
        self.assertEqual(skill_id, "music.skill")
        self.assertAlmostEqual(score, 1.0, places=4)
        self.assertEqual(slots, {})

    def test_bracket_templates_expanded(self):
        p = _make_prototype_pipeline()
        self._register(p, samples=["(play|put on) the music"])
        # two expanded variants -> two prototypes for the label
        n = (p.prototype_store.labels == "music.skill:play_music").sum()
        self.assertEqual(n, 2)

    def test_replacement_on_re_register(self):
        p = _make_prototype_pipeline()
        self._register(p, samples=["one"])
        self._register(p, samples=["two"])  # same triple -> replaces (§8.1)
        self.assertEqual((p.prototype_store.labels == "music.skill:play_music").sum(), 1)

    def test_missing_samples_rejected(self):
        p = _make_prototype_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            self._register(p, samples=[])
        self.assertNotIn("music.skill:play_music", p.intents)
        warn.assert_called()

    def test_missing_identity_rejected(self):
        p = _make_prototype_pipeline()
        msg = Message(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                      data={"intent_name": "x", "samples": ["hi"]})
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            p._handle_intent4_register_template(msg)
        warn.assert_called()
        self.assertEqual(len(p.prototype_store), 0)

    def test_ignored_label_skipped(self):
        p = _make_prototype_pipeline()
        p.ignore_labels = ["music.skill:play_music"]
        self._register(p)
        self.assertEqual(len(p.prototype_store), 0)

    def test_classifier_mode_tracks_label_only(self):
        p = _make_classifier_pipeline()
        msg = Message(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                      data={"skill_id": "music.skill", "intent_name": "play_music",
                            "lang": "en-US", "samples": ["play music"]})
        p._handle_intent4_register_template(msg)
        self.assertIn("music.skill:play_music", p.intents)
        self.assertIsNone(p.prototype_store)


class TestIntent4FrozenClassifierWarning(unittest.TestCase):
    """The frozen classifier warns (once per skill) that it accepted an
    INTENT-4 template registration it can never match; the prototype matcher
    never warns, since it actually consumes the registration."""

    def _msg(self, skill_id="music.skill", intent_name="play_music"):
        return Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={"skill_id": skill_id, "intent_name": intent_name,
                  "lang": "en-US", "samples": ["play music"]},
            context={"skill_id": skill_id},
        )

    def test_classifier_mode_warns_once_per_skill(self):
        p = _make_classifier_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            p._handle_intent4_register_template(self._msg(intent_name="play_music"))
            p._handle_intent4_register_template(self._msg(intent_name="stop_music"))
            p._handle_intent4_register_template(self._msg(skill_id="other.skill"))
        self.assertEqual(warn.call_count, 2)
        first_msg = warn.call_args_list[0].args[0]
        self.assertIn("frozen classifier", first_msg)
        self.assertIn("ovos-m2v-prototype-pipeline", first_msg)
        self.assertIn("music.skill", first_msg)

    def test_prototype_mode_never_warns(self):
        p = _make_prototype_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            p._handle_intent4_register_template(self._msg())
        warn.assert_not_called()


class TestIntent4EntityRegistration(unittest.TestCase):
    def test_entity_fills_template_slots(self):
        p = _make_prototype_pipeline()
        p._handle_intent4_register_entity(Message(
            SpecMessage.ENTITY_REGISTER.value,
            data={"skill_id": "music.skill", "entity_name": "engine",
                  "lang": "en-US", "samples": ["spotify", "youtube"]},
        ))
        self.assertIn("engine", p.entities)
        # register a template that references {engine}
        p._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={"skill_id": "music.skill", "intent_name": "play_on",
                  "lang": "en-US", "samples": ["play on {engine}"]},
        ))
        n = (p.prototype_store.labels == "music.skill:play_on").sum()
        # two entity values -> two filled prototypes
        self.assertEqual(n, 2)

    def test_entity_missing_samples_rejected(self):
        p = _make_prototype_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            p._handle_intent4_register_entity(Message(
                SpecMessage.ENTITY_REGISTER.value,
                data={"skill_id": "s", "entity_name": "engine", "samples": []},
            ))
        self.assertNotIn("engine", p.entities)
        warn.assert_called()

    def test_unregistered_slot_left_literal(self):
        p = _make_prototype_pipeline()
        p._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={"skill_id": "music.skill", "intent_name": "play_on",
                  "lang": "en-US", "samples": ["play on {engine}"]},
        ))
        # no entity registered -> single literal prototype, still accepted
        self.assertIn("music.skill:play_on", p.intents)


class TestIntent4Deregistration(unittest.TestCase):
    def _register(self, p, skill_id="music.skill", intent_name="play_music"):
        p._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={"skill_id": skill_id, "intent_name": intent_name,
                  "lang": "en-US", "samples": ["play music"]},
        ))

    def test_deregister_intent(self):
        p = _make_prototype_pipeline()
        self._register(p)
        p._handle_intent4_deregister_intent(Message(
            SpecMessage.INTENT_DEREGISTER.value,
            data={"skill_id": "music.skill", "intent_name": "play_music", "lang": "en-US"},
        ))
        self.assertNotIn("music.skill:play_music", p.intents)
        self.assertNotIn("music.skill:play_music", p.prototype_store.unique_labels)

    def test_deregister_skill_removes_all(self):
        p = _make_prototype_pipeline()
        self._register(p, intent_name="a")
        self._register(p, intent_name="b")
        self._register(p, skill_id="other.skill", intent_name="c")
        p._handle_intent4_deregister_skill(Message(
            SpecMessage.SKILL_DEREGISTER.value, data={"skill_id": "music.skill"},
        ))
        self.assertNotIn("music.skill:a", p.intents)
        self.assertNotIn("music.skill:b", p.intents)
        self.assertIn("other.skill:c", p.intents)

    def test_deregister_entity(self):
        p = _make_prototype_pipeline()
        p.entities["engine"] = ["spotify"]
        p._handle_intent4_deregister_entity(Message(
            SpecMessage.ENTITY_DEREGISTER.value,
            data={"skill_id": "music.skill", "entity_name": "engine", "lang": "en-US"},
        ))
        self.assertNotIn("engine", p.entities)

    def test_disable_then_enable(self):
        p = _make_prototype_pipeline()
        self._register(p)
        p._handle_intent4_disable(Message(
            SpecMessage.INTENT_DISABLE.value,
            data={"skill_id": "music.skill", "intent_name": "play_music", "lang": "en-US"},
        ))
        self.assertNotIn("music.skill:play_music", p.intents)
        p._handle_intent4_enable(Message(
            SpecMessage.INTENT_ENABLE.value,
            data={"skill_id": "music.skill", "intent_name": "play_music", "lang": "en-US"},
        ))
        self.assertIn("music.skill:play_music", p.intents)


class TestIntent4ContextGating(unittest.TestCase):
    """OVOS-CONTEXT-1 §6/§6.1 requires_context / excludes_context gating."""

    def _register(self, p, requires=None, excludes=None,
                  skill_id="music.skill", intent_name="play_music"):
        data = {"skill_id": skill_id, "intent_name": intent_name,
                "lang": "en-US", "samples": ["play music"]}
        if requires is not None:
            data["requires_context"] = requires
        if excludes is not None:
            data["excludes_context"] = excludes
        p._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value, data=data,
            context={"skill_id": skill_id}))

    def _match_with_context(self, p, intent_context):
        # query embedding identical to the single stored prototype -> cosine 1.0
        p.model.encode.side_effect = None
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        sess = MagicMock()
        sess.intent_context = intent_context
        with patch("ovos_m2v_pipeline.SessionManager.get", return_value=sess):
            return list(p._match("play music"))

    def test_gate_stored_on_register(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["mode"], excludes=[{"key": "busy", "scope": "shared"}])
        self.assertIn("music.skill:play_music", p._context_gates)
        requires, excludes = p._context_gates["music.skill:play_music"]
        self.assertEqual(requires, ["mode"])
        self.assertEqual(excludes, [{"key": "busy", "scope": "shared"}])

    def test_requires_context_present_matches(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["mode"])
        # private key resolves to "<skill_id>:mode"
        results = self._match_with_context(p, {"music.skill:mode": {"value": "party"}})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], "music.skill:play_music")

    def test_requires_context_absent_dropped(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["mode"])
        results = self._match_with_context(p, {})
        self.assertEqual(results, [])

    def test_excludes_context_present_dropped(self):
        p = _make_prototype_pipeline()
        self._register(p, excludes=["busy"])
        results = self._match_with_context(p, {"music.skill:busy": {"value": True}})
        self.assertEqual(results, [])

    def test_excludes_context_absent_matches(self):
        p = _make_prototype_pipeline()
        self._register(p, excludes=["busy"])
        results = self._match_with_context(p, {})
        self.assertEqual(len(results), 1)

    def test_ungated_intent_always_matches(self):
        p = _make_prototype_pipeline()
        self._register(p)  # no requires/excludes
        self.assertNotIn("music.skill:play_music", p._context_gates)
        results = self._match_with_context(p, {})
        self.assertEqual(len(results), 1)

    def test_gate_cleared_on_deregister(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["mode"])
        p._handle_intent4_deregister_intent(Message(
            SpecMessage.INTENT_DEREGISTER.value,
            data={"skill_id": "music.skill", "intent_name": "play_music", "lang": "en-US"}))
        self.assertNotIn("music.skill:play_music", p._context_gates)

    def test_gate_cleared_on_skill_deregister(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["mode"])
        p._handle_intent4_deregister_skill(Message(
            SpecMessage.SKILL_DEREGISTER.value, data={"skill_id": "music.skill"}))
        self.assertNotIn("music.skill:play_music", p._context_gates)


class TestContext1SlotFill(unittest.TestCase):
    """OVOS-CONTEXT-1 §7 context-supplied slots.

    m2v is a label classifier and never extracts a slot value from the
    utterance, so any declared template slot is filled solely from live intent
    context (the "how tall is he" -> ``{person}`` = "Bob" continuous-
    conversation case of spec §3.2). Per the uniform §7 model the fill is
    independent of ``requires_context`` — a declared slot fills from a live
    entry regardless of any gate declaration.
    """

    def _register(self, p, requires=None, samples=None,
                  skill_id="bio.skill", intent_name="height_query"):
        data = {"skill_id": skill_id, "intent_name": intent_name,
                "lang": "en-US",
                "samples": samples if samples is not None else ["how tall is {person}"]}
        if requires is not None:
            data["requires_context"] = requires
        p._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value, data=data,
            context={"skill_id": skill_id}))

    def _match_with_context(self, p, intent_context, utterance="how tall is he"):
        p.model.encode.side_effect = None
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        sess = MagicMock()
        sess.intent_context = intent_context
        sess.blacklisted_intents = []
        sess.blacklisted_skills = []
        with patch("ovos_m2v_pipeline.SessionManager.get", return_value=sess):
            return p.match_high([utterance], "en-US", Message("recognizer_loop:utterance"))

    def test_declared_slot_names_stored_on_register(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=[{"key": "person", "scope": "shared"}])
        self.assertEqual(p._intent_slots.get("bio.skill:height_query"), ["person"])

    def test_slot_names_parsed_before_entity_expansion(self):
        # entity registered for {person}; the stored slot name is still the
        # placeholder, parsed from the original sample, not an expanded value.
        p = _make_prototype_pipeline()
        p._handle_intent4_register_entity(Message(
            SpecMessage.ENTITY_REGISTER.value,
            data={"skill_id": "bio.skill", "entity_name": "person",
                  "lang": "en-US", "samples": ["Alice"]}))
        self._register(p, requires=[{"key": "person", "scope": "shared"}])
        self.assertEqual(p._intent_slots.get("bio.skill:height_query"), ["person"])

    def test_no_slot_no_entry(self):
        p = _make_prototype_pipeline()
        self._register(p, samples=["what time is it"],
                       requires=[{"key": "person", "scope": "shared"}])
        self.assertNotIn("bio.skill:height_query", p._intent_slots)

    def test_context_value_fills_slot_without_requires_context(self):
        # uniform §7: no requires_context declared, yet the declared {person}
        # slot fills from the live shared context entry.
        p = _make_prototype_pipeline()
        self._register(p)  # {person} slot declared, no gate
        self.assertNotIn("bio.skill:height_query", p._context_gates)
        match = self._match_with_context(p, {"person": {"value": "Bob"}})
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("person"), "Bob")

    def test_context_value_fills_slot_with_gate(self):
        # the fill also applies when a requires_context gate is present and
        # satisfied — gate and fill are independent.
        p = _make_prototype_pipeline()
        self._register(p, requires=[{"key": "person", "scope": "shared"}])
        match = self._match_with_context(p, {"person": {"value": "Bob"}})
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("person"), "Bob")

    def test_absent_context_slot_absent(self):
        p = _make_prototype_pipeline()
        self._register(p)  # {person} slot declared, no context entry
        match = self._match_with_context(p, {})
        self.assertIsNotNone(match)
        self.assertNotIn("person", match.match_data)

    def test_flag_only_context_does_not_fill(self):
        p = _make_prototype_pipeline()
        self._register(p)
        match = self._match_with_context(p, {"person": {"value": None}})
        # a null-valued (flag) entry supplies no value to fill the slot
        self.assertIsNotNone(match)
        self.assertNotIn("person", match.match_data)

    def test_slots_cleared_on_deregister(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=[{"key": "person", "scope": "shared"}])
        p._handle_intent4_deregister_intent(Message(
            SpecMessage.INTENT_DEREGISTER.value,
            data={"skill_id": "bio.skill", "intent_name": "height_query",
                  "lang": "en-US"}))
        self.assertNotIn("bio.skill:height_query", p._intent_slots)

    def test_slots_cleared_on_skill_deregister(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=[{"key": "person", "scope": "shared"}])
        p._handle_intent4_deregister_skill(Message(
            SpecMessage.SKILL_DEREGISTER.value, data={"skill_id": "bio.skill"}))
        self.assertNotIn("bio.skill:height_query", p._intent_slots)


class TestIntent4Blacklist(unittest.TestCase):
    """OVOS-INTENT-4 §6.1 template blacklist + session-level blacklists.

    m2v was the only matcher engine that ignored these; the filter mirrors
    padacioso's word-boundary ``_filter`` and adapt/padatious' session
    ``blacklisted_intents`` / ``blacklisted_skills`` gating.
    """

    def _register(self, p, skill_id="music.skill", intent_name="play_music",
                  samples=None, blacklist=None, lang="en-US"):
        data = {
            "skill_id": skill_id,
            "intent_name": intent_name,
            "lang": lang,
            "samples": samples if samples is not None else ["play music"],
        }
        if blacklist is not None:
            data["blacklist"] = blacklist
        p._handle_intent4_register_template(
            Message(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                    data=data, context={"skill_id": skill_id}))

    @staticmethod
    def _pin_query_vector(p):
        # query embedding identical to the stored prototype -> cosine 1.0
        p.model.encode.side_effect = None
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]],
                                               dtype=np.float32)

    def _session_message(self, **session_kwargs):
        from ovos_bus_client.session import Session
        sess = Session(session_id="s")
        for k, v in session_kwargs.items():
            setattr(sess, k, v)
        return Message("recognizer_loop:utterance",
                       context={"session": sess.serialize()})

    def test_blacklist_stored_on_register(self):
        p = _make_prototype_pipeline()
        self._register(p, samples=["play music"], blacklist=["trailer"])
        self.assertEqual(p.excluded_keywords["music.skill:play_music"],
                         ["trailer"])

    def test_blacklist_suppresses_match(self):
        p = _make_prototype_pipeline()
        self._register(p, samples=["play music"], blacklist=["trailer"])
        self._pin_query_vector(p)
        # (a) blacklisted phrase present -> no match (§6.1)
        self.assertEqual(list(p._match("play the trailer")), [])
        # (a) clean utterance still matches
        results = list(p._match("play music"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], "music.skill:play_music")

    def test_session_blacklisted_intent_dropped(self):
        p = _make_prototype_pipeline()
        self._register(p, samples=["play music"])
        self._pin_query_vector(p)
        # control: no blacklist -> matches
        self.assertEqual(len(list(p._match("play music"))), 1)
        # (b) intent blacklisted in session -> dropped
        msg = self._session_message(
            blacklisted_intents=["music.skill:play_music"])
        self.assertEqual(list(p._match("play music", msg)), [])

    def test_session_blacklisted_skill_dropped(self):
        p = _make_prototype_pipeline()
        self._register(p, samples=["play music"])
        self._pin_query_vector(p)
        # (b) skill blacklisted in session -> dropped
        msg = self._session_message(blacklisted_skills=["music.skill"])
        self.assertEqual(list(p._match("play music", msg)), [])

    def test_deregister_intent_drops_blacklist(self):
        p = _make_prototype_pipeline()
        self._register(p, samples=["play music"], blacklist=["trailer"])
        p._handle_intent4_deregister_intent(Message(
            SpecMessage.INTENT_DEREGISTER.value,
            data={"skill_id": "music.skill", "intent_name": "play_music",
                  "lang": "en-US"}))
        self.assertNotIn("music.skill:play_music", p.excluded_keywords)


class TestPadatiousContextGating(unittest.TestCase):
    """OVOS-CONTEXT-1 §6/§6.1 gating for the legacy ``padatious:register_intent``
    wire contract, mirroring ``TestIntent4ContextGating``. §6's tolerated
    ``requires_context`` / ``excludes_context`` extra fields on the padatious
    payload must be honoured the same as on the INTENT-4 template payload."""

    def _register(self, p, requires=None, excludes=None,
                  skill_id="music.skill", intent_name="play_music"):
        data = {"name": f"{skill_id}:{intent_name}.intent",
                "samples": ["play music"]}
        if requires is not None:
            data["requires_context"] = requires
        if excludes is not None:
            data["excludes_context"] = excludes
        p.model.encode.side_effect = None
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        p._handle_register_padatious(Message("padatious:register_intent", data=data))

    def _match_with_context(self, p, intent_context):
        p.model.encode.side_effect = None
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        sess = MagicMock()
        sess.intent_context = intent_context
        with patch("ovos_m2v_pipeline.SessionManager.get", return_value=sess):
            return list(p._match("play music"))

    def test_gate_stored_on_register(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["mode"], excludes=[{"key": "busy", "scope": "shared"}])
        self.assertIn("music.skill:play_music", p._context_gates)
        requires, excludes = p._context_gates["music.skill:play_music"]
        self.assertEqual(requires, ["mode"])
        self.assertEqual(excludes, [{"key": "busy", "scope": "shared"}])

    def test_requires_context_absent_dropped(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["confirming_milk"])
        results = self._match_with_context(p, {})
        self.assertEqual(results, [])

    def test_requires_context_flag_present_matches(self):
        # OVOS-CONTEXT-1 §3.2 confirmation-branch flag shape, verbatim:
        # `{ "value": null, "turns_remaining": 1 }` under the private key
        # `<skill_id>:confirming_milk`.
        p = _make_prototype_pipeline()
        self._register(p, requires=["confirming_milk"])
        results = self._match_with_context(
            p, {"music.skill:confirming_milk": {"value": None, "turns_remaining": 1}})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], "music.skill:play_music")

    def test_requires_context_flag_dead_when_turns_exhausted(self):
        # Same flag shape but `turns_remaining: 0` -> not live per §2 -> gate fails.
        p = _make_prototype_pipeline()
        self._register(p, requires=["confirming_milk"])
        results = self._match_with_context(
            p, {"music.skill:confirming_milk": {"value": None, "turns_remaining": 0}})
        self.assertEqual(results, [])

    def test_excludes_context_present_dropped(self):
        p = _make_prototype_pipeline()
        self._register(p, excludes=["busy"])
        results = self._match_with_context(p, {"music.skill:busy": {"value": None}})
        self.assertEqual(results, [])

    def test_excludes_context_absent_matches(self):
        p = _make_prototype_pipeline()
        self._register(p, excludes=["busy"])
        results = self._match_with_context(p, {})
        self.assertEqual(len(results), 1)

    def test_ungated_sibling_matches_while_gated_excluded(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["mode"], intent_name="play_music")
        self._register(p, intent_name="stop_music")
        results = self._match_with_context(p, {})
        labels = [r[1] for r in results]
        self.assertNotIn("music.skill:play_music", labels)
        self.assertIn("music.skill:stop_music", labels)

    def test_reregistration_replaces_gate(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["mode"])
        self._register(p, requires=None, excludes=["busy"])
        requires, excludes = p._context_gates["music.skill:play_music"]
        self.assertEqual(requires, [])
        self.assertEqual(excludes, ["busy"])

    def test_gate_cleared_on_detach(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["mode"])
        p._handle_detach_intent(Message(
            "detach_intent", data={"intent_name": "music.skill:play_music.intent"}))
        self.assertNotIn("music.skill:play_music", p._context_gates)

    def test_gate_cleared_on_detach_skill(self):
        p = _make_prototype_pipeline()
        self._register(p, requires=["mode"])
        p._handle_detach_skill(Message(
            "detach_skill", data={"skill_id": "music.skill"}))
        self.assertNotIn("music.skill:play_music", p._context_gates)

    def test_context_gate_not_part_of_cache_key(self):
        """The gate is enforced at match time and does not change the
        embeddings, so changing requires_context alone must not re-encode
        (re-hit the prototype cache instead)."""
        cache = MagicMock()
        cache.load.return_value = None
        p = _make_prototype_pipeline()
        p.cache = cache
        p._prototype_cache_enabled = True
        p.prototype_store.cache = cache

        self._register(p, requires=None)
        self.assertEqual(cache.load.call_count, 1)
        key_without_gate = cache.load.call_args[0][1]

        self._register(p, requires=["mode"])
        self.assertEqual(cache.load.call_count, 2)
        key_with_gate = cache.load.call_args[0][1]

        self.assertEqual(key_without_gate, key_with_gate)


if __name__ == "__main__":
    unittest.main()
