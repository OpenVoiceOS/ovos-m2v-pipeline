"""A number slot followed by a noun is filled with numbers that agree with it.

Polish says "za 3 minuty" and "za 5 minut"; "za 3 minut" is not Polish. A
template that writes the noun after the slot has already chosen the number's
agreement class, so a row whose number is outside that class teaches the
classifier a sentence nobody says. The plural rules below are written out from
CLDR here rather than read from the sampler, so the test does not grade the
sampler against itself.
"""
import collections
import sys
from pathlib import Path

import pytest
# Imported, not skipped: a missing parser must fail this file, since every case
# here would otherwise pass with no row checked.
import ovos_spec_tools  # noqa: F401
from ovos_number_parser import pronounce_number

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_from_skills as bfs  # noqa: E402
import typed_slots as ts  # noqa: E402


def _west_slavic(n):
    if n == 1:
        return "one"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "few"
    return "many"


def _east_slavic(n):
    if n % 10 == 1 and n % 100 != 11:
        return "one"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "few"
    return "many"


def _czech(n):
    if n == 1:
        return "one"
    return "few" if 2 <= n <= 4 else "other"


def _one_other(n):
    return "one" if n == 1 else "other"


_english = _one_other


# (language, noun form, the classes it takes, the CLDR rule for the language)
CASES = [
    ("pl-PL", "sekundę", {"one"}, _west_slavic),
    ("pl-PL", "minuty", {"few"}, _west_slavic),
    ("pl-PL", "sekundy", {"few"}, _west_slavic),
    ("pl-PL", "godzin", {"many"}, _west_slavic),
    ("pl-PL", "sekund", {"many"}, _west_slavic),
    ("pl-PL", "dni", {"few", "many"}, _west_slavic),
    ("ru-RU", "минуту", {"one"}, _east_slavic),
    ("ru-RU", "часа", {"few"}, _east_slavic),
    ("ru-RU", "минут", {"many"}, _east_slavic),
    ("uk-UA", "хвилини", {"few"}, _east_slavic),
    ("uk-UA", "годин", {"many"}, _east_slavic),
    ("cs-CZ", "hodinu", {"one"}, _czech),
    ("cs-CZ", "minuty", {"few"}, _czech),
    ("cs-CZ", "sekund", {"other"}, _czech),
    ("en-US", "hour", {"one"}, _english),
    ("en-US", "minutes", {"other"}, _english),
    ("pt-PT", "hora", {"one"}, _one_other),
    ("pt-PT", "horas", {"other"}, _one_other),
    ("pt-BR", "minuto", {"one"}, _one_other),
    ("es-ES", "hora", {"one"}, _one_other),
    ("es-ES", "minutos", {"other"}, _one_other),
    ("ca-ES", "hora", {"one"}, _one_other),
    ("ca-ES", "hores", {"other"}, _one_other),
    ("gl-ES", "horas", {"other"}, _one_other),
    ("fr-FR", "heure", {"one"}, _one_other),
    ("fr-FR", "mois", {"one", "other"}, _one_other),
    ("it-IT", "minuto", {"one"}, _one_other),
    ("sv-SE", "minut", {"one"}, _one_other),
    ("da-DK", "minut", {"one"}, _one_other),
    ("de-DE", "Stunde", {"one"}, _one_other),
]

# Words a number may take before a noun that the parser does not spell itself.
_OVERRIDES = {"une": 1, "un": 1, "una": 1, "en": 1, "et": 1}


def _numbers_in_rows(lang, noun):
    """The number each row of `za {number:offset} <noun> x` carries."""
    from ovos_number_parser.util import GrammaticalGender
    words = dict(_OVERRIDES)
    for n in range(0, 100):
        for gender in GrammaticalGender:
            try:
                words[str(pronounce_number(n, lang=lang, gender=gender))] = n
            except Exception:
                pass
    rows = bfs.fill(f"za {{number:offset}} {noun} x", {}, {},
                    collections.Counter(), lang)
    assert rows, f"{lang} {noun}: no row at all, so this test proves nothing"
    found = []
    for row in rows:
        surface = row[len("za "):-len(f" {noun} x")]
        number = int(surface) if surface.isdigit() else words.get(surface)
        assert number is not None, f"{lang}: cannot read the number in {row!r}"
        found.append((row, number))
    return found


@pytest.mark.parametrize("lang,noun,classes,rule", CASES,
                         ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_every_number_agrees_with_the_noun_after_it(lang, noun, classes, rule):
    for row, number in _numbers_in_rows(lang, noun):
        assert rule(number) in classes, f"{lang}: {row!r} does not agree"


@pytest.mark.parametrize("lang,noun,classes,rule", CASES,
                         ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_a_form_that_serves_several_classes_gets_numbers_from_each(lang, noun, classes, rule):
    """Polish "dni" follows 3 and 5 alike; sampling only one class is a gap."""
    seen = {rule(number) for _, number in _numbers_in_rows(lang, noun)}
    assert seen == classes, (lang, noun, seen)


@pytest.mark.parametrize("row", ["za dwa minuty x", "za dwadzieścia dwa minuty x",
                                 "za czterdzieści dwa minuty x", "za jeden minutę x"])
def test_no_polish_number_word_takes_the_wrong_gender(row):
    """ "dwa minuty" is wrong: a feminine noun takes "dwie". The sampler avoids
    the numbers whose word changes with the noun instead of guessing it."""
    rows = []
    for noun in ("minuty", "minutę"):
        rows += bfs.fill(f"za {{number:offset}} {noun} x", {}, {},
                         collections.Counter(), "pl-PL")
    assert row not in rows, rows


def test_a_word_that_is_not_a_listed_noun_leaves_the_sample_alone():
    """The control: only a listed noun form narrows the sample."""
    stats = collections.Counter()
    rows = bfs.fill("ustaw priorytet na {number:priority}", {}, {}, stats, "pl-PL")
    assert sorted(rows) == sorted(f"ustaw priorytet na {v}"
                                  for v in ts.values_for("number", "pl-PL"))


def test_english_plural_rows_are_unchanged():
    """The control for English: a plural noun keeps the sample it always had."""
    rows = bfs.fill("in {number:offset} minutes", {}, {}, collections.Counter(), "en-US")
    assert sorted(rows) == sorted(f"in {v} minutes"
                                  for v in ts.values_for("number", "en-US"))


def _rows(template, lang):
    return bfs.fill(template, {}, {}, collections.Counter(), lang)


# (language, template, a row that must be there, rows that must not be)
GENDER = [
    ("pt-PT", "em {number:offset} hora x", "em uma hora x", ["em um hora x"]),
    ("pt-PT", "em {number:offset} horas x", "em duas horas x",
     ["em dois horas x", "em quarenta e dois horas x"]),
    ("pt-PT", "em {number:offset} semanas x", "em duas semanas x", ["em dois semanas x"]),
    ("pt-PT", "em {number:offset} minutos x", "em dois minutos x", ["em duas minutos x"]),
    ("pt-BR", "em {number:offset} horas x", "em quarenta e duas horas x",
     ["em quarenta e dois horas x"]),
    ("es-ES", "en {number:offset} hora x", "en una hora x", ["en un hora x", "en uno hora x"]),
    ("es-ES", "en {number:offset} minuto x", "en un minuto x", ["en uno minuto x"]),
    ("ca-ES", "en {number:offset} hores x", "en dues hores x",
     ["en dos hores x", "en quaranta-dos hores x"]),
    ("ca-ES", "en {number:offset} hora x", "en una hora x", ["en un hora x"]),
    ("gl-ES", "en {number:offset} horas x", "en dúas horas x", ["en dous horas x"]),
    ("fr-FR", "dans {number:offset} heure x", "dans une heure x", ["dans un heure x"]),
    ("fr-FR", "dans {number:offset} jour x", "dans un jour x", ["dans une jour x"]),
    ("it-IT", "tra {number:offset} settimana x", "tra una settimana x", ["tra uno settimana x"]),
    ("it-IT", "tra {number:offset} minuto x", "tra un minuto x", ["tra uno minuto x"]),
    ("sv-SE", "om {number:offset} minut x", "om en minut x", ["om ett minut x"]),
    ("da-DK", "om {number:offset} minut x", "om et minut x", ["om en minut x"]),
]


@pytest.mark.parametrize("lang,template,wanted,unwanted", GENDER,
                         ids=[f"{g[0]}-{g[2]}" for g in GENDER])
def test_the_number_word_takes_the_gender_of_the_noun(lang, template, wanted, unwanted):
    rows = _rows(template, lang)
    assert wanted in rows, rows
    for row in unwanted:
        assert row not in rows, rows


@pytest.mark.parametrize("row", ["tra una ora x", "tra uno ora x", "tra un ora x"])
def test_an_elided_italian_one_is_left_to_the_digits(row):
    """Italian writes "un'ora", which a slot followed by a space cannot."""
    rows = _rows("tra {number:offset} ora x", "it-IT")
    assert "tra 1 ora x" in rows, rows
    assert row not in rows, rows


@pytest.mark.parametrize("row", ["in eins Stunde x", "in eine Stunde x", "in einer Stunde x"])
def test_a_german_one_is_left_to_the_digits(row):
    """German "ein" changes with the case the preposition takes."""
    rows = _rows("in {number:offset} Stunde x", "de-DE")
    assert "in 1 Stunde x" in rows, rows
    assert row not in rows, rows


@pytest.mark.parametrize("noun,wrong", [("минуту", "один"), ("минута", "один"),
                                        ("час", "одна"), ("неделю", "один")])
def test_a_russian_one_never_takes_the_wrong_gender(noun, wrong):
    """ "один минуту" is wrong; the word for one changes with gender and case,
    which the parser does not give, so the digits carry it."""
    rows = _rows(f"через {{number:offset}} {noun} x", "ru-RU")
    assert f"через 1 {noun} x" in rows, rows
    assert f"через {wrong} {noun} x" not in rows, rows
    assert not any(row.startswith(("через один ", "через одна ", "через одну "))
                   for row in rows), rows


def test_a_form_no_number_takes_drops_the_sentence():
    """ "3 часы" is never Russian: no cardinal takes the nominative plural."""
    stats = collections.Counter()
    rows = bfs.fill("когда это {number:offset} часы x", {}, {}, stats, "ru-RU")
    assert rows == []
    assert stats["sentence_dropped_noun_never_follows_a_number"] == 1
    assert stats["sentence_dropped_no_typed_values_for_this_language"] == 0


def test_the_correct_forms_beside_it_still_fill():
    """The control: only the form no number takes is dropped."""
    rows = _rows("когда это {number:offset} (часы|часа|часов) x", "ru-RU")
    assert rows and not any("часы" in row for row in rows), rows
    assert any("часа" in row for row in rows) and any("часов" in row for row in rows)


def test_a_template_of_only_that_form_is_reported_apart_from_the_unexplained():
    outcomes = [("ru-RU", {"number"}, "когда это {number:offset} часы", []),
                ("ru-RU", {"number"}, "через {number:offset} часа", ["через 3 часа"])]
    dead, lost, gaps, never = bfs.typed_slot_template_losses(outcomes)
    assert lost == [] and gaps == [] and dead == []
    assert never == ["ru-RU number когда это {number:offset} часы"]
