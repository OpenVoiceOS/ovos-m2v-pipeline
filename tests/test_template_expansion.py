"""Malformed-template tolerance during intent registration (OVOS-INTENT-4).

Real-world skill locale files contain malformed template lines
(translated slot names such as ``{Medien}``, truncated braces such as
``{location``, adjacent slots such as ``{a}{b}``). A consuming plugin
skips each such template with a WARN carrying the §5.3 fields
(skill_id, intent_name, lang, topic, reason) and indexes the remaining
valid templates; the registration is rejected only when no valid
template remains. Registrations are per-lang messages, so a rejected
registration in one lang never affects another lang's registrations.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message

from ovos_m2v_pipeline import _parse_intent_file

MALFORMED = ["play {Medien}", "weather in {location", "{first}{second}"]
VALID = ["play {media}", "(start|begin) {media}", "[UNUSED] noise"]
FIELDS = ["'test_skill'", "en-US"]


def _make_prototype_pipeline():
    config = {"model": "fake-embed-model", "mode": "prototype"}

    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value = np.zeros((1, 4), dtype=np.float32)

    fake_m2v = MagicMock()
    fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch.dict(sys.modules, {"model2vec": fake_m2v}):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)

    pipeline.model = mock_embed_model
    pipeline.prototype_store = MagicMock()
    pipeline.prototype_store.add.return_value = 1
    return pipeline


def _padatious_msg(samples, name="test_skill:demo.intent",
                   skill_id="test_skill", lang="en-US"):
    return Message("padatious:register_intent",
                   data={"name": name, "samples": samples,
                         "skill_id": skill_id, "lang": lang})


class TestParseIntentFile(unittest.TestCase):
    def _write(self, tmp, lines):
        path = os.path.join(tmp, "demo.intent")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return path

    def test_malformed_lines_skipped_valid_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, MALFORMED + VALID)
            with patch("ovos_m2v_pipeline.LOG.warning") as warn:
                sentences = _parse_intent_file(path, "[ctx]")
        self.assertIn("play {media}", sentences)
        self.assertIn("start {media}", sentences)
        self.assertIn("begin {media}", sentences)
        self.assertIn("noise", sentences)
        self.assertNotIn("play {Medien}", sentences)
        self.assertNotIn("weather in {location", sentences)
        self.assertNotIn("{first}{second}", sentences)
        # one WARN per skipped template, carrying the caller's ctx fields
        self.assertEqual(warn.call_count, len(MALFORMED))
        for call in warn.call_args_list:
            self.assertIn("skipping malformed template", call[0][0])
            self.assertIn("[ctx]", call[0][0])

    def test_all_malformed_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, MALFORMED)
            with patch("ovos_m2v_pipeline.LOG.warning"):
                self.assertEqual(_parse_intent_file(path), [])


class TestRegisterPadatious(unittest.TestCase):
    def test_mixed_samples_register_valid_ones_with_field_warns(self):
        pipeline = _make_prototype_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            pipeline._handle_register_padatious(
                _padatious_msg(MALFORMED + VALID))
        self.assertIn("test_skill:demo", pipeline.intents)
        pipeline.prototype_store.add.assert_called_once()
        sentences = pipeline.prototype_store.add.call_args[0][2]
        self.assertIn("play {media}", sentences)
        self.assertNotIn("play {Medien}", sentences)
        self.assertEqual(warn.call_count, len(MALFORMED))
        for call in warn.call_args_list:
            log = call[0][0]
            self.assertIn("skipping malformed template", log)
            for field in FIELDS + ["test_skill:demo",
                                   "padatious:register_intent"]:
                self.assertIn(field, log)

    def test_all_malformed_rejects_registration_with_warn(self):
        pipeline = _make_prototype_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            pipeline._handle_register_padatious(_padatious_msg(list(MALFORMED)))
        self.assertNotIn("test_skill:demo", pipeline.intents)
        pipeline.prototype_store.add.assert_not_called()
        rejection = warn.call_args_list[-1][0][0]
        self.assertIn("rejecting registration", rejection)
        self.assertIn("no valid template remains", rejection)
        for field in FIELDS:
            self.assertIn(field, rejection)

    def test_malformed_file_skips_lines_keeps_valid(self):
        pipeline = _make_prototype_pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "demo.intent")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(MALFORMED + VALID))
            msg = Message("padatious:register_intent",
                          data={"name": "test_skill:demo",
                                "skill_id": "test_skill", "lang": "en-US",
                                "file_name": path})
            with patch("ovos_m2v_pipeline.LOG.warning") as warn:
                pipeline._handle_register_padatious(msg)
        self.assertIn("test_skill:demo", pipeline.intents)
        pipeline.prototype_store.add.assert_called_once()
        for call in warn.call_args_list:
            for field in FIELDS:
                self.assertIn(field, call[0][0])

    def test_rejected_lang_does_not_affect_other_langs(self):
        # each registration message is per-lang: an all-malformed de-DE
        # registration must not block the valid en-US one
        pipeline = _make_prototype_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning"):
            pipeline._handle_register_padatious(
                _padatious_msg(["spiele {Medien}"], lang="de-DE"))
        pipeline.prototype_store.add.assert_not_called()
        pipeline._handle_register_padatious(
            _padatious_msg(list(VALID), lang="en-US"))
        self.assertIn("test_skill:demo", pipeline.intents)
        pipeline.prototype_store.add.assert_called_once()


class TestIntent4Registration(unittest.TestCase):
    def _template_msg(self, samples, lang="en-US"):
        return Message("ovos.intent.register.template",
                       data={"skill_id": "test_skill",
                             "intent_name": "demo",
                             "lang": lang, "samples": samples})

    def test_malformed_template_skipped_valid_indexed(self):
        pipeline = _make_prototype_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            pipeline._handle_intent4_register_template(
                self._template_msg(MALFORMED + VALID))
        pipeline.prototype_store.add.assert_called_once()
        expanded = pipeline.prototype_store.add.call_args[0][2]
        self.assertIn("play {media}", expanded)
        self.assertNotIn("play {Medien}", expanded)
        self.assertEqual(warn.call_count, len(MALFORMED))
        for call in warn.call_args_list:
            log = call[0][0]
            self.assertIn("skipping malformed template", log)
            for field in FIELDS + ["'demo'"]:
                self.assertIn(field, log)

    def test_all_malformed_rejected_with_warn(self):
        pipeline = _make_prototype_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            pipeline._handle_intent4_register_template(
                self._template_msg(list(MALFORMED)))
        pipeline.prototype_store.add.assert_not_called()
        rejection = warn.call_args_list[-1][0][0]
        self.assertIn("rejecting", rejection)
        for field in FIELDS + ["'demo'"]:
            self.assertIn(field, rejection)

    def test_malformed_entity_sample_skipped_valid_indexed(self):
        pipeline = _make_prototype_pipeline()
        msg = Message("ovos.entity.register",
                      data={"skill_id": "test_skill",
                            "entity_name": "media",
                            "lang": "en-US",
                            "samples": ["spotify", "{bad"]})
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            pipeline._handle_intent4_register_entity(msg)
        self.assertEqual(pipeline.entities.get("media"), ["spotify"])
        warn.assert_called_once()
        self.assertIn("skipping malformed entity sample", warn.call_args[0][0])

    def test_all_malformed_entity_samples_rejected(self):
        pipeline = _make_prototype_pipeline()
        msg = Message("ovos.entity.register",
                      data={"skill_id": "test_skill",
                            "entity_name": "media",
                            "lang": "en-US",
                            "samples": ["{bad", "{Worse}"]})
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            pipeline._handle_intent4_register_entity(msg)
        self.assertNotIn("media", pipeline.entities)
        rejection = warn.call_args_list[-1][0][0]
        self.assertIn("rejecting", rejection)
        self.assertIn("no valid entity sample remains", rejection)


if __name__ == "__main__":
    unittest.main()
