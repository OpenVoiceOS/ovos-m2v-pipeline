"""The laugh renames must be bridged, vocabularies excluded.

`ovos-skill-laugh` renames two intents and two vocabulary files. Only the
intents are labels a model emits, so only they get a pair.
"""
import re

from ovos_m2v_pipeline.renames import RENAMED_LABELS

SKILL = "ovos-skill-laugh.openvoiceos"


def test_both_renamed_intents_have_a_pair():
    assert RENAMED_LABELS[f"{SKILL}:Laugh"] == f"{SKILL}:laugh"
    assert RENAMED_LABELS[f"{SKILL}:RandomLaugh"] == f"{SKILL}:random_laugh"


def test_the_vocabulary_renames_get_no_pair():
    """Laugh.voc and Stop.voc are vocabularies; no model emits them."""
    assert f"{SKILL}:Stop" not in RENAMED_LABELS


def test_the_adapt_intent_name_is_not_a_resource():
    """StopLaughing is a Python identifier in an IntentBuilder, not a file."""
    assert f"{SKILL}:StopLaughing" not in RENAMED_LABELS


def test_every_target_stays_spec_compliant():
    for target in RENAMED_LABELS.values():
        assert re.match(r"^[a-z0-9_]+$", target.partition(":")[2]), target
