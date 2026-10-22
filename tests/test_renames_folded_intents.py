"""Folded intents are bridged to their survivor, whatever spelling a model emits.

ovos-skill-date-time#476 folds weekday_matches_date and what_weekday_is_it
into weekday_for_date, and ovos-skill-icanhazdadjokes#186 folds search_joke
into joke. A published model emits the dotted or the underscored old name.
"""
import pytest

from ovos_m2v_pipeline.renames import RENAMED_LABELS

DATE_TIME = "ovos-skill-date-time.openvoiceos"
JOKES = "ovos-skill-icanhazdadjokes.openvoiceos"

FOLDS = [
    (DATE_TIME, ["weekday.for.date", "weekday.matches.date", "weekday_matches_date",
                 "what.weekday.is.it", "what_weekday_is_it"], "weekday_for_date"),
    (JOKES, ["search_joke"], "joke"),
]


@pytest.mark.parametrize("skill,olds,survivor", FOLDS)
def test_every_old_spelling_names_the_survivor(skill, olds, survivor):
    for old in olds:
        assert RENAMED_LABELS[f"{skill}:{old}"] == f"{skill}:{survivor}", old


@pytest.mark.parametrize("skill,olds,survivor", FOLDS)
def test_the_survivor_is_not_itself_a_source(skill, olds, survivor):
    assert f"{skill}:{survivor}" not in RENAMED_LABELS

