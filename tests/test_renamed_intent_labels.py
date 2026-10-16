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


class TestSpeedtestRename(unittest.TestCase):
    def test_speedtest_rename_is_listed(self):
        self.assertEqual(
            RENAMED_LABELS["ovos-skill-speedtest.openvoiceos:SpeedtestIntent"],
            "ovos-skill-speedtest.openvoiceos:speedtest_intent")


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
        self.assertEqual(len(dated), 14)

    def test_confucius_quotes_rename_is_listed(self):
        self.assertEqual(
            RENAMED_LABELS["ovos-skill-confucius-quotes.openvoiceos:ConfuciusQuote"],
            "ovos-skill-confucius-quotes.openvoiceos:confucius_quote")

    def test_no_new_label_is_also_an_old_label(self):
        # the alias is applied once; a chained rename would stop half way
        chained = set(RENAMED_LABELS.values()) & set(RENAMED_LABELS)
        self.assertEqual(chained, set())


def _builder():
    pytest.importorskip("pandas")
    pytest.importorskip("yaml")
    pytest.importorskip("ovos_spec_tools")
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "train" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("build_dataset_under_test", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def _resolve(builder, label, registered):
    """``resolve_label`` against a registry holding exactly *registered*."""
    by_skill = {}
    for name in registered:
        skill, _, intent = name.partition(":")
        by_skill.setdefault(skill, set()).add(intent)
    by_fold = {builder.fold_skill(skill): [skill] for skill in by_skill}
    return builder.resolve_label(label, set(registered), by_fold, by_skill, {})


def test_builder_files_pre_rename_rows_under_the_new_label():
    builder = _builder()
    for intent in ("what.time.is.it",
                   # an older spelling that INTENT_ALIASES folds into the dotted name
                   "handle_show_time"):
        label = builder.make_label("ovos-skill-date-time", intent)
        assert _resolve(builder, label, {NEW})[0] == NEW, intent


DATE = "ovos-skill-date-time.openvoiceos:current_date"
DAY = "ovos-skill-date-time.openvoiceos:what_day_is_it"


@pytest.mark.parametrize("intent", ["current_date", "what.date.is.it",
                                    "handle_query_date_simple"])
def test_builder_follows_the_fold_only_where_the_pin_made_it(intent):
    """date-time#466 folded current_date into what_day_is_it. A pin before it
    registers both, and its current_date rows keep their label; a pin after
    it registers only the survivor, and the rows move there."""
    builder = _builder()
    label = builder.make_label("ovos-skill-date-time", intent)
    assert _resolve(builder, label, {DATE, DAY})[0] == DATE
    assert _resolve(builder, label, {DAY})[0] == DAY


def test_builder_leaves_a_rename_alone_when_the_pin_lacks_both_names():
    """A pin that registers neither name has nothing to rename to, so the
    label must not move to the survivor it does not register."""
    builder = _builder()
    label = builder.make_label("ovos-skill-date-time", "current_date")
    other = {"ovos-skill-date-time.openvoiceos:what_year_is_it"}
    assert _resolve(builder, label, other)[0] != DAY


def test_days_in_history_rename_is_listed():
    assert RENAMED_LABELS["ovos-skill-days-in-history.openvoiceos:TellMeMoreIntent"] == \
        "ovos-skill-days-in-history.openvoiceos:tell_me_more_intent"
