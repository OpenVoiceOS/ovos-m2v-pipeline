"""The alerts rename wave must be bridged completely, not partly.

`ovos-skill-alerts` renames 19 intent resources to OVOS-INTENT-2 §2 names.
A model trained before the rename emits the old label, so each old label
needs a pair here or that intent stops routing the moment the skill ships.

A half-filled table is the failure that matters: the skill renames 19 names
and the table carries 17, and the two that are missing fail silently at
runtime. This test names all 19.
"""
import re

from ovos_m2v_pipeline.renames import RENAMED_LABELS

SKILL = "ovos-skill-alerts.openvoiceos"

RENAMED = {
    "AddListSubitems": "add_list_subitems",
    "CalendarList": "calendar_list",
    "CancelAlert": "cancel_alert",
    "ChangeMediaProperties": "change_media_properties",
    "ChangePriority": "change_priority",
    "ChangeRepeat": "change_repeat",
    "ChangeUntil": "change_until",
    "CreateAlarm": "create_alarm",
    "CreateAlarmAlt": "create_alarm_alt",
    "CreateEvent": "create_event",
    "CreateList": "create_list",
    "CreateReminder": "create_reminder",
    "CreateTimer": "create_timer",
    "DAVSync": "dav_sync",
    "DeleteList": "delete_list",
    "ListAlerts": "list_alerts",
    "QueryListNames": "query_list_names",
    "RescheduleAlert": "reschedule_alert",
    "TimerStatus": "timer_status",
}

COMPLIANT = re.compile(r"^[a-z0-9_]+$")


def test_every_renamed_alerts_intent_has_a_pair():
    missing = [old for old in RENAMED if f"{SKILL}:{old}" not in RENAMED_LABELS]
    assert not missing, f"no bridge for {missing}"


def test_each_pair_points_at_the_name_the_skill_registers_now():
    for old, new in RENAMED.items():
        assert RENAMED_LABELS[f"{SKILL}:{old}"] == f"{SKILL}:{new}"


def test_every_target_in_the_table_is_spec_compliant():
    """The table must never point at a name §2 refuses, for any skill."""
    for target in RENAMED_LABELS.values():
        intent = target.partition(":")[2]
        assert COMPLIANT.match(intent), target


def test_no_pair_points_at_itself():
    """A pair that maps a label to itself hides a rename that never happened."""
    for old, new in RENAMED_LABELS.items():
        assert old != new, old


def test_the_table_has_one_target_per_source():
    """The control: a source label resolves once, not through a chain."""
    for target in RENAMED_LABELS.values():
        assert target not in RENAMED_LABELS, f"{target} is both a target and a source"
