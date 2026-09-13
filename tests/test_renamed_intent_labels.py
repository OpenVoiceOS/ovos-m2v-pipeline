"""A published model keeps routing to an intent whose skill renamed it.

A skill that renames an intent resource file registers the new name. A model
trained before the rename still emits the old label. Classifier mode keeps
only model classes the running skills register, so without an alias the old
label is dropped and the renamed intent can no longer be reached through the
classifier. ``ovos_m2v_pipeline.renames.RENAMED_LABELS`` maps the old label to
the new one at inference and when the training corpus is built.
"""
import re
import unittest

import pytest

from ovos_m2v_pipeline.renames import RENAMED_LABELS
from tests.test_pipeline import _make_pipeline, _setup_model

OLD = "ovos-skill-date-time.openvoiceos:what.time.is.it"
NEW = "ovos-skill-date-time.openvoiceos:what_time_is_it"


class TestRenamedLabelAtInference(unittest.TestCase):
    def test_old_model_label_routes_to_the_renamed_intent(self):
        # the skill registers only the new name; the model emits the old one
        p = _make_pipeline(intents=[NEW], renormalize=False)
        _setup_model(p, [OLD], [0.9])
        results = list(p._match("what time is it"))
        self.assertEqual(len(results), 1, "the old model label was dropped")
        skill_id, label, prob, _ = results[0]
        self.assertEqual(skill_id, "ovos-skill-date-time.openvoiceos")
        self.assertEqual(label, NEW)
        self.assertAlmostEqual(prob, 0.9)

    def test_skill_that_still_registers_the_old_name_keeps_it(self):
        # a pre-rename install: the alias must not send it to a name it lacks
        p = _make_pipeline(intents=[OLD], renormalize=False)
        _setup_model(p, [OLD], [0.9])
        results = list(p._match("what time is it"))
        self.assertEqual([r[1] for r in results], [OLD])

    def test_no_alias_when_neither_name_is_registered(self):
        p = _make_pipeline(intents=["other.skill:other"], renormalize=False)
        _setup_model(p, [OLD, "other.skill:other"], [0.6, 0.4])
        labels = [r[1] for r in p._match("what time is it")]
        self.assertNotIn(NEW, labels)
        self.assertNotIn(OLD, labels)

    def test_user_label_map_for_the_old_label_turns_the_rename_off(self):
        # the user maps the old label themselves: the rename must not apply
        p = _make_pipeline(config={"model": "fake",
                                   "label_map": {OLD: "custom.skill:custom.time"}},
                           intents=[NEW], renormalize=False)
        _setup_model(p, [OLD], [0.9])
        self.assertNotIn(NEW, [r[1] for r in p._match("what time is it")])

    def test_user_label_map_for_the_old_label_still_applies(self):
        # as for any label_map entry, the raw label must be registered to match
        p = _make_pipeline(config={"model": "fake",
                                   "label_map": {OLD: "custom.skill:custom.time"}},
                           intents=[OLD, NEW], renormalize=False)
        _setup_model(p, [OLD], [0.9])
        results = list(p._match("what time is it"))
        self.assertEqual(len(results), 1)
        skill_id, label, _, _ = results[0]
        self.assertEqual((skill_id, label), ("custom.skill", "custom.skill:custom.time"))

    def test_valid_labels_is_still_checked_against_the_raw_model_label(self):
        p = _make_pipeline(config={"model": "fake", "valid_labels": [OLD]},
                           intents=[NEW], renormalize=False)
        _setup_model(p, [OLD], [0.9])
        self.assertEqual([r[1] for r in p._match("what time is it")], [NEW])


class TestRenameTable(unittest.TestCase):
    def test_every_entry_keeps_its_skill_and_names_a_compliant_intent(self):
        legal = re.compile(r"^[a-z0-9_]+$")  # OVOS-INTENT-2 §2
        for old, new in RENAMED_LABELS.items():
            old_skill, _, _ = old.partition(":")
            new_skill, _, new_intent = new.partition(":")
            self.assertEqual(old_skill, new_skill, old)
            self.assertRegex(new_intent, legal, new)
            self.assertNotEqual(old, new)

    def test_date_time_renames_are_listed(self):
        dated = {k for k in RENAMED_LABELS if k.startswith("ovos-skill-date-time.openvoiceos:")}
        self.assertEqual(len(dated), 13)


def test_builder_files_pre_rename_rows_under_the_new_label():
    pytest.importorskip("pandas")
    pytest.importorskip("yaml")
    pytest.importorskip("ovos_spec_tools")
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "train" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("build_dataset_under_test", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.make_label("ovos-skill-date-time", "what.time.is.it") == NEW
    # an older spelling that INTENT_ALIASES folds into the dotted name
    assert builder.make_label("ovos-skill-date-time", "handle_show_time") == NEW


def test_days_in_history_rename_is_listed():
    assert RENAMED_LABELS["ovos-skill-days-in-history.openvoiceos:TellMeMoreIntent"] == \
        "ovos-skill-days-in-history.openvoiceos:tell_me_more_intent"
