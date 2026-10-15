"""Every label the published model emits for a folded skill reaches an intent
the skill's dev head registers.

``OpenVoiceOS/ovos-m2v-intents-multilingual`` at 145a1bd was trained before
ovos-skill-alerts and ovos-skill-date-time folded several intents into their
siblings. The skills no longer register the folded names, so the classifier
keeps such a class only if ``RENAMED_LABELS`` routes it to a registered name.
The rename lookup is one step: an old name has to point straight at the
survivor, not at an intermediate name the skill also dropped.

Both sides are frozen here. ``PUBLISHED`` is the model's ``labels.json``
``valid_labels`` for the two skills; ``DEV_HEAD_INTENTS`` is what
``train/build_dataset.registered_intents`` reads from each skill at the sha
named beside it. Neither derives from ``RENAMED_LABELS``.
"""
from tests.test_pipeline import _make_pipeline

ALERTS = "ovos-skill-alerts.openvoiceos"
DATE_TIME = "ovos-skill-date-time.openvoiceos"

#: ``valid_labels`` of OpenVoiceOS/ovos-m2v-intents-multilingual@145a1bd94acb,
#: restricted to the two skills.
PUBLISHED = {
    ALERTS: {
        "AddListSubitems", "CalendarList", "CancelAlert",
        "ChangeMediaProperties", "ChangeProperties", "CreateAlarm",
        "CreateEvent", "CreateList", "CreateOcpAlarm", "CreateReminder",
        "CreateTimer", "DAVSync", "DeleteList", "DeleteListEntries",
        "DeleteTodoEntries", "ListAlerts", "QueryListEntries",
        "QueryListNames", "QueryTodoEntries", "RescheduleAlert",
        "TimerStatus", "create_reminder_recurring", "missed_alerts",
    },
    DATE_TIME: {
        "current_date", "date.future.weekend", "date.last.weekend",
        "is.leap.year", "next.leap.year", "time.until", "weekday.for.date",
        "weekday.matches.date", "what.day.is.it", "what.month.is.it",
        "what.time.is.it", "what.time.will.it.be", "what.weekday.is.it",
        "what.year.is.it",
    },
}

#: Intents each skill registers at its dev head: ovos-skill-alerts 68e59b73,
#: ovos-skill-date-time 1b559790.
DEV_HEAD_INTENTS = {
    ALERTS: {
        "add_list_subitems", "calendar_list", "cancel_alert",
        "change_media_properties", "change_priority", "change_repeat",
        "change_until", "create_alarm", "create_event", "create_list",
        "create_reminder", "create_timer", "dav_sync", "delete_list",
        "delete_list_entries", "list_alerts", "missed_alerts",
        "query_list_entries", "query_list_names", "reschedule_alert",
        "timer_status",
    },
    DATE_TIME: {
        "date_future_weekend", "date_last_weekend", "is_leap_year",
        "next_leap_year", "time_until", "weekday_for_date",
        "weekday_matches_date", "what_day_is_it", "what_month_is_it",
        "what_time_is_it", "what_time_will_it_be", "what_weekday_is_it",
        "what_year_is_it",
    },
}

#: Published alerts labels with no route. ChangeProperties names a family
#: that alerts#234 relabelled across change_priority, change_repeat and
#: change_until, so no single route is right. alerts#216 renamed the list
#: entry intents and merged the todo-kind ones into them, and alerts#199 and
#: #270 folded CreateOcpAlarm into create_alarm; those five have a survivor
#: and no entry yet. The test below fails when one gains a route or another
#: label loses one.
UNROUTED = {f"{ALERTS}:{n}" for n in (
    "ChangeProperties", "CreateOcpAlarm", "DeleteListEntries",
    "DeleteTodoEntries", "QueryListEntries", "QueryTodoEntries")}


def _labels(table):
    return {f"{skill}:{name}" for skill, names in table.items() for name in names}


def _routes():
    registered = _labels(DEV_HEAD_INTENTS)
    pipeline = _make_pipeline(intents=registered, renormalize=False)
    renamed = pipeline._renamed_labels_to_route()
    return registered, {label: renamed.get(label, label)
                        for label in sorted(_labels(PUBLISHED))}


def test_every_published_label_routes_to_a_registered_intent():
    registered, routes = _routes()
    unrouted = {label for label, target in routes.items()
                if target not in registered}
    assert unrouted == UNROUTED, {
        "routes to nothing": {label: routes[label]
                              for label in sorted(unrouted - UNROUTED)},
        "now routed": sorted(UNROUTED - unrouted)}


def test_the_folded_intents_route_to_their_survivor():
    _, routes = _routes()
    assert routes[f"{ALERTS}:create_reminder_recurring"] == f"{ALERTS}:create_reminder"
    assert routes[f"{DATE_TIME}:current_date"] == f"{DATE_TIME}:what_day_is_it"


def test_both_alarm_alt_spellings_route_to_create_alarm():
    """Neither spelling is a class of this model, so they are checked with
    the dev head's registrations directly."""
    pipeline = _make_pipeline(intents=_labels(DEV_HEAD_INTENTS), renormalize=False)
    renamed = pipeline._renamed_labels_to_route()
    assert renamed[f"{ALERTS}:CreateAlarmAlt"] == f"{ALERTS}:create_alarm"
    assert renamed[f"{ALERTS}:create_alarm_alt"] == f"{ALERTS}:create_alarm"


def test_a_skill_that_still_registers_current_date_keeps_it():
    """The date-time pin registers current_date beside what_day_is_it; the
    fold must not move it there."""
    intents = _labels(DEV_HEAD_INTENTS) | {f"{DATE_TIME}:current_date"}
    pipeline = _make_pipeline(intents=intents, renormalize=False)
    assert f"{DATE_TIME}:current_date" not in pipeline._renamed_labels_to_route()
