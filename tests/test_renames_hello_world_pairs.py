"""The hello-world renames must be bridged, one pair per intent label.

`ovos-skill-hello-world#140` renames its four intents to OVOS-INTENT-2 §2
names. A model trained before the rename emits the old labels, so each
gets a pair until a model trained on the new names is published. The two
`.voc` and two `.dialog` files the same commit renamed get no pair: no
model emits them as a label.
"""
import re

from ovos_m2v_pipeline.renames import RENAMED_LABELS

SKILL = "ovos-skill-hello-world.openvoiceos"
PAIRS = {
    "Greetings": "greetings",
    "HelloWorldIntent": "hello_world_intent",
    "HowAreYou": "how_are_you",
    "ThankYouIntent": "thank_you_intent",
}


def test_every_old_label_has_its_pair():
    for old, new in PAIRS.items():
        assert RENAMED_LABELS[f"{SKILL}:{old}"] == f"{SKILL}:{new}", old


def test_no_new_label_is_itself_a_source():
    """The control: a target must not be re-mapped by another entry."""
    for new in PAIRS.values():
        assert f"{SKILL}:{new}" not in RENAMED_LABELS, new


def test_the_vocabularies_and_dialogs_get_no_pair():
    for name in ("HelloWorldKeyword", "ThankYouKeyword", "hello.world", "how.are.you"):
        assert f"{SKILL}:{name}" not in RENAMED_LABELS, name


def test_every_target_stays_spec_compliant():
    for target in RENAMED_LABELS.values():
        assert re.match(r"^[a-z0-9_]+$", target.partition(":")[2]), target
