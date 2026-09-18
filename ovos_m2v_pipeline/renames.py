"""Intent labels a skill renamed after a published model was trained.

A skill that renames a resource file registers a new intent name. A model
trained before the rename still emits the old label, and a label no skill
registers can never match. This table maps the old label to the one the
skill registers now.

It is read twice:

* at runtime, by ``Model2VecIntentPipeline``, as a ``label_map`` layer, so a
  published model keeps routing to the renamed intent;
* by ``train/build_dataset.py``, so corpus rows filed under the old spelling
  train the new label.

Keys and values are full ``<skill_id>:<intent_name>`` labels, as the skill
registers them on the bus (see ``docs/labels.md``). The module has no imports,
so the builder can read it without the pipeline's runtime dependencies.
"""

#: old label (as the model emits it) -> label the skill registers now
RENAMED_LABELS = {
    # ovos-skill-days-in-history: resource base names made OVOS-INTENT-2 §2
    # compliant (lowercase letters, digits and underscores).
    "ovos-skill-days-in-history.openvoiceos:TellMeMoreIntent": "ovos-skill-days-in-history.openvoiceos:tell_me_more_intent",
    # ovos-skill-date-time: resource base names made OVOS-INTENT-2 §2
    # compliant (lowercase letters, digits and underscores).
    "ovos-skill-date-time.openvoiceos:date.future.weekend": "ovos-skill-date-time.openvoiceos:date_future_weekend",
    "ovos-skill-date-time.openvoiceos:date.last.weekend": "ovos-skill-date-time.openvoiceos:date_last_weekend",
    "ovos-skill-date-time.openvoiceos:is.leap.year": "ovos-skill-date-time.openvoiceos:is_leap_year",
    "ovos-skill-date-time.openvoiceos:next.leap.year": "ovos-skill-date-time.openvoiceos:next_leap_year",
    "ovos-skill-date-time.openvoiceos:time.until": "ovos-skill-date-time.openvoiceos:time_until",
    # ovos-skill-speedtest: resource base names made OVOS-INTENT-2 §2
    # compliant (lowercase letters, digits and underscores).
    "ovos-skill-speedtest.openvoiceos:SpeedtestIntent": "ovos-skill-speedtest.openvoiceos:speedtest_intent",
    "ovos-skill-date-time.openvoiceos:weekday.for.date": "ovos-skill-date-time.openvoiceos:weekday_for_date",
    "ovos-skill-date-time.openvoiceos:weekday.matches.date": "ovos-skill-date-time.openvoiceos:weekday_matches_date",
    "ovos-skill-date-time.openvoiceos:what.day.is.it": "ovos-skill-date-time.openvoiceos:what_day_is_it",
    "ovos-skill-date-time.openvoiceos:what.month.is.it": "ovos-skill-date-time.openvoiceos:what_month_is_it",
    "ovos-skill-date-time.openvoiceos:what.time.is.it": "ovos-skill-date-time.openvoiceos:what_time_is_it",
    "ovos-skill-date-time.openvoiceos:what.time.will.it.be": "ovos-skill-date-time.openvoiceos:what_time_will_it_be",
    "ovos-skill-date-time.openvoiceos:what.weekday.is.it": "ovos-skill-date-time.openvoiceos:what_weekday_is_it",
    "ovos-skill-date-time.openvoiceos:what.year.is.it": "ovos-skill-date-time.openvoiceos:what_year_is_it",
    # ovos-skill-confucius-quotes: resource base names made OVOS-INTENT-2 §2
    # compliant (lowercase letters, digits and underscores).
    "ovos-skill-confucius-quotes.openvoiceos:ConfuciusQuote": "ovos-skill-confucius-quotes.openvoiceos:confucius_quote",
    # ovos-skill-ip: resource base names made OVOS-INTENT-2 §2 compliant
    # (lowercase letters, digits and underscores).
    "ovos-skill-ip.openvoiceos:IPIntent": "ovos-skill-ip.openvoiceos:ip",
    "ovos-skill-ip.openvoiceos:LastIPDigitsIntent": "ovos-skill-ip.openvoiceos:last_ip_digits",
    "ovos-skill-ip.openvoiceos:PublicIPIntent": "ovos-skill-ip.openvoiceos:public_ip",
    "ovos-skill-ip.openvoiceos:what.ssid": "ovos-skill-ip.openvoiceos:what_ssid",
    # ovos-skill-alerts: resource base names made OVOS-INTENT-2 §2
    # compliant (lowercase letters, digits and underscores).
    "ovos-skill-alerts.openvoiceos:AddListSubitems": "ovos-skill-alerts.openvoiceos:add_list_subitems",
    "ovos-skill-alerts.openvoiceos:CalendarList": "ovos-skill-alerts.openvoiceos:calendar_list",
    "ovos-skill-alerts.openvoiceos:CancelAlert": "ovos-skill-alerts.openvoiceos:cancel_alert",
    "ovos-skill-alerts.openvoiceos:ChangeMediaProperties": "ovos-skill-alerts.openvoiceos:change_media_properties",
    "ovos-skill-alerts.openvoiceos:ChangePriority": "ovos-skill-alerts.openvoiceos:change_priority",
    "ovos-skill-alerts.openvoiceos:ChangeRepeat": "ovos-skill-alerts.openvoiceos:change_repeat",
    "ovos-skill-alerts.openvoiceos:ChangeUntil": "ovos-skill-alerts.openvoiceos:change_until",
    "ovos-skill-alerts.openvoiceos:CreateAlarm": "ovos-skill-alerts.openvoiceos:create_alarm",
    "ovos-skill-alerts.openvoiceos:CreateAlarmAlt": "ovos-skill-alerts.openvoiceos:create_alarm_alt",
    "ovos-skill-alerts.openvoiceos:CreateEvent": "ovos-skill-alerts.openvoiceos:create_event",
    "ovos-skill-alerts.openvoiceos:CreateList": "ovos-skill-alerts.openvoiceos:create_list",
    "ovos-skill-alerts.openvoiceos:CreateReminder": "ovos-skill-alerts.openvoiceos:create_reminder",
    "ovos-skill-alerts.openvoiceos:CreateTimer": "ovos-skill-alerts.openvoiceos:create_timer",
    "ovos-skill-alerts.openvoiceos:DAVSync": "ovos-skill-alerts.openvoiceos:dav_sync",
    "ovos-skill-alerts.openvoiceos:DeleteList": "ovos-skill-alerts.openvoiceos:delete_list",
    "ovos-skill-alerts.openvoiceos:ListAlerts": "ovos-skill-alerts.openvoiceos:list_alerts",
    "ovos-skill-alerts.openvoiceos:QueryListNames": "ovos-skill-alerts.openvoiceos:query_list_names",
    "ovos-skill-alerts.openvoiceos:RescheduleAlert": "ovos-skill-alerts.openvoiceos:reschedule_alert",
    "ovos-skill-alerts.openvoiceos:TimerStatus": "ovos-skill-alerts.openvoiceos:timer_status",
    # ovos-skill-wikipedia: WikiMore.intent renamed to wiki_more.intent
    # (OVOS-INTENT-2 §2); the duplicate copies of both were removed.
    "ovos-skill-wikipedia.openvoiceos:WikiMore": "ovos-skill-wikipedia.openvoiceos:wiki_more",
}
