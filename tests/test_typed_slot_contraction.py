"""A number slot after a word that joins with the number is filled so they stay apart.

Catalan elides "de" before a vowel sound ("d'una hora", "d'1 hora"). A slot
written after "de" as a separate token cannot produce the elision, so after
"de" nothing that starts with the sound of un, una, 1 or 11 is kept, and
after "d'" only that is.
"""
import collections
import re
import sys
from pathlib import Path

import pytest
# Imported, not skipped: a missing parser must fail this file, since every case
# here would otherwise pass with no row checked.
import ovos_spec_tools  # noqa: F401
import ovos_number_parser  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_from_skills as bfs  # noqa: E402


def _rows(template, lang, stats=None):
    return bfs.fill(template, {}, {}, stats if stats is not None else collections.Counter(),
                    lang)


@pytest.mark.parametrize("template,word", [
    ("que hora será dentro de {number:offset} hora", "dentro de unha hora"),
    ("que hora será en {number:offset} minuto", "en un minuto"),
    ("cal será o ano con {number:offset} día", "con un día"),
])
def test_galician_keeps_the_uncontracted_word(template, word):
    """Galician is not in the table: the skill accepts "con un" beside "cun"."""
    rows = _rows(template, "gl-ES")
    assert any(row.endswith(word) for row in rows), rows


def test_catalan_de_before_a_singular_noun_has_no_row():
    """Only un, una and 1 take "hora", and all three elide after "de"."""
    stats = collections.Counter()
    rows = _rows("quina hora serà després de {number:offset} hora", "ca-ES", stats)
    assert rows == [], rows
    assert stats["sentence_dropped_number_joins_the_word_before"] == 1


def test_catalan_de_before_a_plural_noun_keeps_its_numbers():
    rows = _rows("quina hora serà després de {number:offset} hores", "ca-ES")
    assert "quina hora serà després de dues hores" in rows, rows
    assert not any(re.search(r"\bde (?:1|11|un|una|onze)\b", row) for row in rows), rows


def test_catalan_elided_d_takes_only_what_elides():
    singular = _rows("quina hora serà després d'{number:offset} hora", "ca-ES")
    assert sorted(singular) == ["quina hora serà després d'1 hora",
                                "quina hora serà després d'una hora"], singular
    plural = _rows("quina hora serà després d'{number:offset} hores", "ca-ES")
    assert plural == [], plural


def test_other_languages_are_unchanged():
    """Portuguese "de um" and Spanish "de una" are standard and left alone."""
    assert "dentro de una hora" in _rows("dentro de {number:offset} hora", "es-ES")


_PRODUCING = ("ca-ES", {"number"}, "skill ca-ES: després de {number:offset} hores", 6)


def test_a_template_with_no_row_for_this_reason_is_reported_apart():
    outcomes = [("ca-ES", {"number"}, "skill ca-ES: després de {number:offset} hora", 0),
                _PRODUCING]
    dead, lost, gaps, never, joined = bfs.typed_slot_template_losses(outcomes)
    assert lost == [] and gaps == [] and dead == [] and never == []
    assert joined == ["ca-ES number skill ca-ES: després de {number:offset} hora"]


@pytest.mark.parametrize("template", [
    "skill ca-ES: més de {number:offset} persones",
    "skill ca-ES: després de {number:offset} hores a {location}",
])
def test_a_loss_the_word_before_did_not_cause_stays_unexplained(template):
    """"de" is listed, but it leaves these slots with numbers, so their
    missing rows have another cause and the build must still refuse them."""
    outcomes = [("ca-ES", {"number"}, template, 0), _PRODUCING]
    dead, lost, gaps, never, joined = bfs.typed_slot_template_losses(outcomes)
    assert joined == [], joined
    assert lost == [f"number {template}"], lost


def test_alternation_before_the_slot_is_read_as_the_word_fill_sees():
    template = "skill ca-ES: quina hora serà (després de|abans de) {number:offset} hora"
    outcomes = [("ca-ES", {"number"}, template, 0), _PRODUCING]
    dead, lost, gaps, never, joined = bfs.typed_slot_template_losses(outcomes)
    assert lost == [], lost
    assert joined == [f"ca-ES number {template}"], joined
    stats = collections.Counter()
    assert _rows(template.split(": ", 1)[1], "ca-ES", stats) == []
    assert stats["sentence_dropped_number_joins_the_word_before"] == 2
