"""The ip rename wave must be bridged completely.

`ovos-skill-ip` renames three CamelCase intents and one dotted intent to
OVOS-INTENT-2 §2 names. A model trained before the rename emits the old
label, so each needs a pair here or that intent stops routing the moment the
skill ships.

The dotted name is the one worth naming twice: `what.ssid` carries a dot in
its base name, and the runtime strips only the last suffix, so the old label
and the new one are two different strings to every reader.
"""
import re

from ovos_m2v_pipeline.renames import RENAMED_LABELS

SKILL = "ovos-skill-ip.openvoiceos"

RENAMED = {
    "IPIntent": "ip",
    "LastIPDigitsIntent": "last_ip_digits",
    "PublicIPIntent": "public_ip",
    "what.ssid": "what_ssid",
}

COMPLIANT = re.compile(r"^[a-z0-9_]+$")


def test_every_renamed_ip_intent_has_a_pair():
    missing = [old for old in RENAMED if f"{SKILL}:{old}" not in RENAMED_LABELS]
    assert not missing, f"no bridge for {missing}"


def test_each_pair_points_at_the_name_the_skill_registers_now():
    for old, new in RENAMED.items():
        assert RENAMED_LABELS[f"{SKILL}:{old}"] == f"{SKILL}:{new}"


def test_the_dotted_source_survives_as_written():
    """A dot in the source key is the whole point of that entry."""
    assert f"{SKILL}:what.ssid" in RENAMED_LABELS


def test_every_target_in_the_table_is_spec_compliant():
    for target in RENAMED_LABELS.values():
        assert COMPLIANT.match(target.partition(":")[2]), target


def test_no_dialog_only_rename_is_in_the_table():
    """The table holds intent labels, never dialog names.

    `ethernet.connection`, `my.public.ip`, `public.ip.error` and
    `wifi.signal` are dialog files. A dialog name is never a label a model
    emits, so a pair for one would be noise that outlives the wave.
    """
    for dialog in ("ethernet.connection", "my.public.ip", "public.ip.error",
                   "wifi.signal"):
        assert f"{SKILL}:{dialog}" not in RENAMED_LABELS, dialog
