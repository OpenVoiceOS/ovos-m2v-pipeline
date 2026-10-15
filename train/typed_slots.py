"""Locale-correct surface values for a typed slot, at dataset build time.

A typed slot is written `{number:offset}` (OVOS-INTENT-1 §5.6). The runtime
binds it by asking a parser for the span it covers, so the corpus must carry
the SURFACE a speaker of that language would say, never the literal
`{number:offset}`. A row that keeps the placeholder teaches the classifier a
brace.

The parsers are the source of truth, one per type, the same libraries the
runtime transformer uses:

* number, `ovos_number_parser.pronounce_number`
* date and duration, `ovos_date_parser`
* color, `ovos_color_parser`
* language, `ovos_lang_parser.pronounce_lang`

Which types exist at all is read, never re-declared: `REGISTERED_TYPES` in
`ovos_spec_tools` is the registry, and `TypedSlotsTransformer.supported_types`
in `ovos-typed-slots-transformer` is what the runtime resolves. Those two
sets are NOT equal today, and the difference is deliberate here: a type the
runtime cannot resolve is reported, not sampled, because a training row the
runtime can never reproduce is worse than a row that is missing.

A type with no generator for a language yields nothing rather than an
English value. A German row that says "forty two" is not German data.

A parser that is not installed is a different answer, and it is never
silence: `values_for` raises `MissingParser`. An absent package empties every
language at once, so reading it as "no generator for this language" drops
every row of that type from the corpus while the build reports success.
The packages are the `corpus` extra of this project.
"""
from __future__ import annotations

import datetime
import functools
import importlib
import re
from typing import Dict, FrozenSet, List

SAMPLES_PER_SLOT = 3


class MissingParser(RuntimeError):
    """A parser the typed-slot sampler reads is not installed."""


def _require(module: str):
    """Import *module*, or raise `MissingParser` naming its package."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        package = module.split(".")[0].replace("_", "-")
        raise MissingParser(
            f"{module} is not importable ({exc}). Typed slots are filled from "
            f"{package}; without it every row with such a slot is dropped. "
            f"Install the corpus extra: uv pip install 'ovos-m2v-pipeline[corpus]'"
        ) from exc

#: How many surface forms one sampled value yields, by type. A numeric value
#: has two surfaces: the DIGITS an ASR front end sends ("in 2 hours") and the
#: WORDS a person says ("in two hours"). The corpus carries both, because the
#: classifier meets both. Words alone leave no row that matches what the
#: recogniser emits, which is the state this entry ends (orchestrator ruling,
#: 2026-09-28, on the en-US row loss measured for v7.0).
SURFACES_PER_VALUE = {"number": 2}


def cap_for(slot_type: str) -> int:
    """How many surfaces one slot of this type may contribute.

    The cap bounds corpus growth and it counts SURFACES, not values. A type
    with two surfaces per value therefore keeps SAMPLES_PER_SLOT values, not
    half of them.
    """
    return SAMPLES_PER_SLOT * SURFACES_PER_VALUE.get(slot_type, 1)


#: Anchors chosen to be unremarkable in every calendar: a mid-month day in a
#: non-leap year, and durations a person would actually say.
_DATE_ANCHORS = [
    datetime.datetime(2024, 3, 15, 9, 30),
    datetime.datetime(2024, 7, 4, 18, 0),
    datetime.datetime(2024, 11, 22, 12, 0),
]
_NUMBERS = [3, 15, 42]
_DURATIONS = [300, 3600, 5400]


def registry() -> FrozenSet[str]:
    """Every type the specification registers."""
    from ovos_spec_tools import REGISTERED_TYPES
    return frozenset(REGISTERED_TYPES)


def resolvable() -> FrozenSet[str]:
    """Every type the runtime transformer binds, read from the plugin.

    The plugin is required. Without it the sampler cannot tell which types
    the runtime binds, and sampling a type it cannot bind ships rows the
    runtime never reproduces.
    """
    plugin = _require("ovos_typed_slots_transformer")
    return frozenset(plugin.TypedSlotsTransformer.supported_types)


def unresolvable() -> FrozenSet[str]:
    """Registered types the runtime does not bind. Reported, never sampled."""
    return registry() - resolvable()


def _numbers(lang: str) -> List[str]:
    """Both surfaces of each anchor: the digits, then the language's words.

    The digits are language-neutral and are emitted for every language, so a
    language whose wordlist `ovos_number_parser` lacks still gets numeric
    rows instead of none. The word form is still that language's own: a
    German row does not say "forty two". The parser package itself is
    required; without it this raises `MissingParser` rather than shipping a
    digits-only corpus nobody chose.

    The two surfaces of one anchor sit together, digits first, and `cap_for`
    admits both surfaces of every anchor, so nothing is truncated here.
    """
    pronounce_number = _require("ovos_number_parser").pronounce_number
    out = []
    for value in _NUMBERS:
        out.append(str(value))
        try:
            spoken = pronounce_number(value, lang=lang)
        except Exception:
            continue
        if spoken and "{" not in spoken:
            out.append(str(spoken))
    return out


#: Numeral-noun agreement. The noun after a number takes a form the number
#: decides: Portuguese "em 1 hora" but "em 3 horas", Polish "za 3 minuty" and
#: "za 22 minuty" but "za 5 minut". In languages with grammatical gender the
#: number word also agrees with the noun: "uma hora" and "duas horas" but "um
#: minuto" and "dois minutos", Catalan "dues hores". A template that writes a
#: noun right after a number slot therefore fixes which numbers fit it and how
#: their words are spelled, and a number drawn without looking at that noun
#: produces "em 3 hora" or "em quarenta e dois horas".
#:
#: Each language lists its `nouns`, each a gender and its form per agreement
#: class; a form that serves two classes (Polish "dni" after 3 and after 5)
#: is written under both. `values` holds the numbers sampled for each class;
#: where "two" changes with gender (Portuguese, Catalan, Galician) the plural
#: sample starts at 2, so both spellings reach the corpus.
#: The classes follow the CLDR cardinal plural rules for integers.
#:
#: The number word comes from `ovos_number_parser` in the noun's gender, with
#: `gender` naming the parser's gender for each of the table's. `words`
#: overrides the parser where it has no gendered form: French "une heure",
#: Swedish "en minut". `digits_only` names the classes whose number words
#: also change with case ("jedną minutę", "in einer Stunde"), which no source
#: here gives; those rows carry the digits alone. Where the parser has no
#: gendered form either and the word does change, the class `values` avoid
#: such numbers: Polish says "dwa dni" but "dwie minuty", so no number ending
#: in 2 is sampled there.
AGREEMENT = {
    "en": {
        "values": {"one": (1,), "other": (3, 15, 42)},
        "nouns": [
            ("", {"one": "second", "other": "seconds"}),
            ("", {"one": "minute", "other": "minutes"}),
            ("", {"one": "hour", "other": "hours"}),
            ("", {"one": "day", "other": "days"}),
            ("", {"one": "week", "other": "weeks"}),
            ("", {"one": "month", "other": "months"}),
            ("", {"one": "year", "other": "years"}),
        ],
    },
    "pt": {
        "values": {"one": (1,), "other": (2, 15, 42)},
        "gender": {"m": "masculine", "f": "feminine"},
        "nouns": [
            ("m", {"one": "segundo", "other": "segundos"}),
            ("m", {"one": "minuto", "other": "minutos"}),
            ("f", {"one": "hora", "other": "horas"}),
            ("m", {"one": "dia", "other": "dias"}),
            ("f", {"one": "semana", "other": "semanas"}),
            ("m", {"one": "mês", "other": "meses"}),
            ("m", {"one": "ano", "other": "anos"}),
        ],
    },
    # Before a noun Spanish shortens "uno" to "un" ("un minuto", "veintiún
    # minutos"); the parser calls that form neutral.
    "es": {
        "values": {"one": (1,), "other": (3, 15, 42)},
        "gender": {"m": "neutral", "f": "feminine"},
        "nouns": [
            ("m", {"one": "segundo", "other": "segundos"}),
            ("m", {"one": "minuto", "other": "minutos"}),
            ("f", {"one": "hora", "other": "horas"}),
            ("m", {"one": "día", "other": "días"}),
            ("f", {"one": "semana", "other": "semanas"}),
            ("m", {"one": "mes", "other": "meses"}),
            ("m", {"one": "año", "other": "años"}),
        ],
    },
    # Aragonese feminine "two" varies by dialect ("dos", "das"), so no number
    # ending in 2 is sampled.
    "an": {
        "values": {"one": (1,), "other": (3, 15, 30)},
        "gender": {"m": "masculine", "f": "feminine"},
        "nouns": [
            ("m", {"one": "segundo", "other": "segundos"}),
            ("m", {"one": "minuto", "other": "minutos"}),
            ("f", {"one": "hora", "other": "horas"}),
        ],
    },
    "ca": {
        "values": {"one": (1,), "other": (2, 15, 42)},
        "gender": {"m": "masculine", "f": "feminine"},
        "nouns": [
            ("m", {"one": "segon", "other": "segons"}),
            ("m", {"one": "minut", "other": "minuts"}),
            ("f", {"one": "hora", "other": "hores"}),
            ("m", {"one": "dia", "other": "dies"}),
            ("f", {"one": "setmana", "other": "setmanes"}),
            ("m", {"one": "mes", "other": "mesos"}),
            ("m", {"one": "any", "other": "anys"}),
        ],
    },
    "gl": {
        "values": {"one": (1,), "other": (2, 15, 42)},
        "gender": {"m": "masculine", "f": "feminine"},
        "nouns": [
            ("m", {"one": "segundo", "other": "segundos"}),
            ("m", {"one": "minuto", "other": "minutos"}),
            ("f", {"one": "hora", "other": "horas"}),
            ("m", {"one": "día", "other": "días"}),
            ("f", {"one": "semana", "other": "semanas"}),
            ("m", {"one": "mes", "other": "meses"}),
            ("m", {"one": "ano", "other": "anos"}),
        ],
    },
    "fr": {
        "values": {"one": (1,), "other": (3, 15, 42)},
        "words": {("f", 1): "une"},
        "nouns": [
            ("f", {"one": "seconde", "other": "secondes"}),
            ("f", {"one": "minute", "other": "minutes"}),
            ("f", {"one": "heure", "other": "heures"}),
            ("m", {"one": "jour", "other": "jours"}),
            ("f", {"one": "semaine", "other": "semaines"}),
            ("m", {"one": "mois", "other": "mois"}),
            ("m", {"one": "an", "other": "ans"}),
        ],
    },
    # Italian writes "un minuto" and "una settimana", but "un'ora": the
    # elision joins the word to the noun, which a slot cannot write, so the
    # feminine "one" before a vowel is left to the digits.
    "it": {
        "values": {"one": (1,), "other": (3, 15, 42)},
        "words": {("m", 1): "un", ("f", 1): "una", ("fv", 1): None},
        "nouns": [
            ("m", {"one": "secondo", "other": "secondi"}),
            ("m", {"one": "minuto", "other": "minuti"}),
            ("fv", {"one": "ora", "other": "ore"}),
            ("m", {"one": "giorno", "other": "giorni"}),
            ("f", {"one": "settimana", "other": "settimane"}),
            ("m", {"one": "mese", "other": "mesi"}),
            ("m", {"one": "anno", "other": "anni"}),
        ],
    },
    "de": {
        "values": {"one": (1,), "other": (3, 15, 42)},
        "digits_only": ("one",),
        "nouns": [
            ("f", {"one": "sekunde", "other": "sekunden"}),
            ("f", {"one": "minute", "other": "minuten"}),
            ("f", {"one": "stunde", "other": "stunden"}),
            ("m", {"one": "tag", "other": "tage"}),
            ("f", {"one": "woche", "other": "wochen"}),
            ("m", {"one": "monat", "other": "monate"}),
            ("n", {"one": "jahr", "other": "jahre"}),
        ],
    },
    "sv": {
        "values": {"one": (1,), "other": (3, 15, 42)},
        "words": {("c", 1): "en", ("n", 1): "ett"},
        "nouns": [
            ("c", {"one": "sekund", "other": "sekunder"}),
            ("c", {"one": "minut", "other": "minuter"}),
            ("c", {"one": "timme", "other": "timmar"}),
            ("c", {"one": "dag", "other": "dagar"}),
            ("c", {"one": "vecka", "other": "veckor"}),
            ("c", {"one": "månad", "other": "månader"}),
            ("n", {"one": "år", "other": "år"}),
        ],
    },
    "da": {
        "values": {"one": (1,), "other": (3, 15, 42)},
        "words": {("c", 1): "en", ("n", 1): "et"},
        "nouns": [
            ("n", {"one": "sekund", "other": "sekunder"}),
            ("n", {"one": "minut", "other": "minutter"}),
            ("c", {"one": "time", "other": "timer"}),
            ("c", {"one": "dag", "other": "dage"}),
            ("c", {"one": "uge", "other": "uger"}),
            ("c", {"one": "måned", "other": "måneder"}),
            ("n", {"one": "år", "other": "år"}),
        ],
    },
    "pl": {
        "values": {"one": (1,), "few": (3, 4, 23), "many": (5, 15, 30)},
        "digits_only": ("one",),
        "nouns": [
            ("f", {"one": "sekundę", "few": "sekundy", "many": "sekund"}),
            ("f", {"one": "sekunda"}),
            ("f", {"one": "minutę", "few": "minuty", "many": "minut"}),
            ("f", {"one": "minuta"}),
            ("f", {"one": "godzinę", "few": "godziny", "many": "godzin"}),
            ("f", {"one": "godzina"}),
            ("m", {"one": "dzień", "few": "dni", "many": "dni"}),
            ("m", {"one": "tydzień", "few": "tygodnie", "many": "tygodni"}),
            ("m", {"one": "miesiąc", "few": "miesiące", "many": "miesięcy"}),
            ("m", {"one": "rok", "few": "lata", "many": "lat"}),
        ],
    },
    # "часы" is the nominative plural, which no cardinal takes ("3 часа",
    # "5 часов"); a template that writes it after a number has no right row.
    "ru": {
        "values": {"one": (1, 21, 31), "few": (3, 4, 23), "many": (5, 15, 30)},
        "digits_only": ("one",),
        "never": ("часы",),
        "nouns": [
            ("f", {"one": "секунду", "few": "секунды", "many": "секунд"}),
            ("f", {"one": "секунда"}),
            ("f", {"one": "минуту", "few": "минуты", "many": "минут"}),
            ("f", {"one": "минута"}),
            ("m", {"one": "час", "few": "часа", "many": "часов"}),
            ("m", {"one": "день", "few": "дня", "many": "дней"}),
            ("f", {"one": "неделю", "few": "недели", "many": "недель"}),
            ("f", {"one": "неделя"}),
            ("m", {"one": "месяц", "few": "месяца", "many": "месяцев"}),
            ("m", {"one": "год", "few": "года", "many": "лет"}),
        ],
    },
    "uk": {
        "values": {"one": (1, 21, 31), "few": (3, 4, 23), "many": (5, 15, 30)},
        "digits_only": ("one",),
        "nouns": [
            ("f", {"one": "секунду", "few": "секунди", "many": "секунд"}),
            ("f", {"one": "секунда"}),
            ("f", {"one": "хвилину", "few": "хвилини", "many": "хвилин"}),
            ("f", {"one": "хвилина"}),
            ("f", {"one": "годину", "few": "години", "many": "годин"}),
            ("f", {"one": "година"}),
            ("m", {"one": "день", "few": "дні", "many": "днів"}),
            ("m", {"one": "тиждень", "few": "тижні", "many": "тижнів"}),
            ("m", {"one": "місяць", "few": "місяці", "many": "місяців"}),
            ("m", {"one": "рік", "few": "роки", "many": "років"}),
        ],
    },
    "cs": {
        "values": {"one": (1,), "few": (3, 4), "other": (5, 15, 30)},
        "digits_only": ("one",),
        "nouns": [
            ("f", {"one": "sekundu", "few": "sekundy", "other": "sekund"}),
            ("f", {"one": "sekunda"}),
            ("f", {"one": "minutu", "few": "minuty", "other": "minut"}),
            ("f", {"one": "minuta"}),
            ("f", {"one": "hodinu", "few": "hodiny", "other": "hodin"}),
            ("f", {"one": "hodina"}),
            ("m", {"one": "den", "few": "dny", "other": "dní"}),
            ("m", {"other": "dnů"}),
            ("m", {"one": "týden", "few": "týdny", "other": "týdnů"}),
            ("m", {"one": "měsíc", "few": "měsíce", "other": "měsíců"}),
            ("m", {"one": "rok", "few": "roky", "other": "let"}),
        ],
    },
}


@functools.lru_cache(maxsize=None)
def _forms(lang: str) -> Dict[str, tuple]:
    """Each noun form of a language, as (its agreement classes, its gender)."""
    out = {}
    for gender, by_class in AGREEMENT[lang]["nouns"]:
        for cls, form in by_class.items():
            classes, _ = out.get(form, ((), gender))
            out[form] = (classes + (cls,), gender)
    return out


def agreement(lang: str, noun: str) -> tuple:
    """(classes, gender) for a noun form after a number, or ((), "").

    No classes means this language has no agreement table or the word is not
    a noun form it lists, and the number is sampled as it always was.
    """
    key = lang.split("-")[0].lower()
    if key not in AGREEMENT or not noun:
        return (), ""
    return _forms(key).get(noun.lower(), ((), ""))


#: Words before a number slot that join with some of the number's surfaces.
#: Catalan elides "de" before a vowel sound, and 1 and 11 are read "u" and
#: "onze": "d'una hora", "d'1 hora". A slot written after "de" as a separate
#: word cannot produce the elision, so after "de" no such surface is kept and
#: the sentence is dropped when nothing else is left; after "d'" only such a
#: surface is kept. Each entry says whether matching surfaces are dropped or
#: are the only ones kept. The pattern reads the start of a surface, so a
#: grouped "1.000" or "1 000" would also count as eliding; no such number is
#: sampled.
_CA_ELIDES = r"^(?:1|11|un|una|onze)(?:\b|$)"
BEFORE_A_NUMBER = {
    "ca": {
        "de": ("drop", _CA_ELIDES),
        "d'": ("keep", _CA_ELIDES),
        "d\u2019": ("keep", _CA_ELIDES),
    },
}


def joins_the_word_before(lang: str, word: str) -> bool:
    """Whether *word*, written before a number slot, restricts its surfaces."""
    table = BEFORE_A_NUMBER.get(lang.split("-")[0].lower(), {})
    return bool(word) and word.lower() in table


def emptied_by_the_word_before(lang: str, noun: str, before: str) -> bool:
    """Whether the word before a number slot alone leaves it with no surface.

    True only when the slot has numbers without that word and none with it,
    so a slot that is empty for any other reason is not explained by it.
    """
    if not joins_the_word_before(lang, before):
        return False
    return (not values_for("number", lang, noun, before)
            and bool(values_for("number", lang, noun)))


def _fits_after(values, lang: str, word: str) -> list:
    """The surfaces of *values* that can stand after *word*."""
    table = BEFORE_A_NUMBER.get(lang.split("-")[0].lower(), {})
    rule = table.get(word.lower()) if word else None
    if rule is None:
        return list(values)
    mode, pattern = rule
    matches = [re.match(pattern, v, flags=re.IGNORECASE) is not None for v in values]
    keep = (lambda m: not m) if mode == "drop" else (lambda m: m)
    return [v for v, m in zip(values, matches) if keep(m)]


def never_after_a_number(lang: str, noun: str) -> bool:
    """Whether *noun* is a form no number in this language can stand before."""
    table = AGREEMENT.get(lang.split("-")[0].lower(), {})
    return bool(noun) and noun.lower() in table.get("never", ())


def _number_word(value: int, lang: str, gender: str):
    """The word for *value* before a noun of *gender*, or None for digits only."""
    table = AGREEMENT[lang.split("-")[0].lower()]
    words = table.get("words", {})
    if (gender, value) in words:
        return words[(gender, value)]
    number_parser = _require("ovos_number_parser")
    parser_gender = table.get("gender", {}).get(gender)
    if parser_gender is None:
        return number_parser.pronounce_number(value, lang=lang)
    grammatical = _require("ovos_number_parser.util").GrammaticalGender
    return number_parser.pronounce_number(value, lang=lang,
                                          gender=grammatical(parser_gender))


def _agreeing_numbers(lang: str, classes: tuple, gender: str) -> List[str]:
    """Surfaces of the numbers that agree with a noun of these classes.

    The classes are taken in turn, one number from each, so a form that
    serves two classes gets numbers from both rather than only the first.
    """
    table = AGREEMENT[lang.split("-")[0].lower()]
    pools = [list(table["values"][c]) for c in classes]
    order = []
    while any(pools):
        for pool in pools:
            if pool:
                order.append(pool.pop(0))
    digits_only = {v for c in table.get("digits_only", ()) for v in table["values"][c]}
    out = []
    for value in order[:SAMPLES_PER_SLOT]:
        out.append(str(value))
        if value in digits_only:
            continue
        try:
            spoken = _number_word(value, lang, gender)
        except MissingParser:
            raise
        except Exception:
            continue
        if spoken and "{" not in spoken:
            out.append(str(spoken))
    return out


def _dates(lang: str) -> List[str]:
    nice_date = _require("ovos_date_parser").nice_date
    out = []
    for anchor in _DATE_ANCHORS:
        try:
            spoken = nice_date(anchor, lang=lang)
        except Exception:
            continue
        if spoken:
            out.append(str(spoken))
    return out


def _durations(lang: str) -> List[str]:
    nice_duration = _require("ovos_date_parser").nice_duration
    out = []
    for seconds in _DURATIONS:
        try:
            spoken = nice_duration(seconds, lang=lang)
        except Exception:
            continue
        if spoken:
            out.append(str(spoken))
    return out


#: Primaries, as RGB. `lookup_name` names a COLOUR, so the sample set is
#: three colours rather than three English words translated.
_COLOURS = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]


def _colors(lang: str) -> List[str]:
    """Colour names for a language, or nothing when the language has none.

    `ovos_color_parser` names a colour in the language asked for, so the
    sample is the language's own word: `Rouge` for fr, `Vermell` for ca. A
    language with no term set yields nothing, because an English colour name
    in a Finnish row is not Finnish data.
    """
    lookup_name = _require("ovos_color_parser").lookup_name
    sRGBAColor = _require("ovos_color_parser.models").sRGBAColor
    out = []
    for red, green, blue in _COLOURS:
        try:
            spoken = lookup_name(sRGBAColor(red, green, blue), lang=lang)
        except Exception:
            continue
        if spoken and isinstance(spoken, str):
            out.append(spoken)
    return out


#: Three languages a user names often, as codes. `pronounce_lang` names the
#: LANGUAGE in the language asked for, so the sample is the locale's own
#: word: `Francès` for ca, `Tafransist` for kab.
_LANGUAGE_CODES = ["en", "fr", "de"]


def _languages(lang: str) -> List[str]:
    """Language names for a language, or nothing when the parser has none.

    `ovos_lang_parser` raises ValueError for a language with no bundled
    wordlist (the same behaviour the runtime transformer handles), and that
    is the empty answer here: an English "French" in a Finnish row is not
    Finnish data.
    """
    pronounce_lang = _require("ovos_lang_parser").pronounce_lang
    out = []
    for code in _LANGUAGE_CODES:
        try:
            spoken = pronounce_lang(code, lang=lang)
        except Exception:
            continue
        if spoken and isinstance(spoken, str):
            out.append(spoken)
    return out


_GENERATORS = {
    "number": _numbers,
    "date": _dates,
    "duration": _durations,
    "color": _colors,
    "language": _languages,
}


@functools.lru_cache(maxsize=None)
def values_for(slot_type: str, lang: str, noun: str = "", before: str = "") -> tuple:
    """Up to `cap_for(slot_type)` surfaces for one type in one language.

    *noun* is the word the template writes right after the slot. For a
    number followed by a noun form listed in `AGREEMENT`, only numbers that
    agree with that form are sampled, and their words take the noun's
    gender. A form no number can stand before yields nothing, so the
    sentence is dropped. Any other word changes nothing.

    *before* is the word written right before the slot. A word listed in
    `BEFORE_A_NUMBER` keeps only the surfaces that can follow it as a
    separate word, which may leave nothing.

    An empty result is a real answer: this language has no generator for
    this type, and the caller must leave the template out rather than fill
    it with another language's words. A parser that is not installed is not
    that answer, and raises `MissingParser`.
    """
    if slot_type not in resolvable():
        return ()
    generator = _GENERATORS.get(slot_type)
    if generator is None:
        return ()
    if slot_type == "number" and never_after_a_number(lang, noun):
        return ()
    classes, gender = agreement(lang, noun) if slot_type == "number" else ((), "")
    try:
        produced = (_agreeing_numbers(lang, classes, gender) if classes
                    else generator(lang))
    except MissingParser:
        raise
    except Exception:
        return ()
    if slot_type == "number":
        produced = _fits_after(produced, lang, before)
    seen, out = set(), []
    for value in produced:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
        if len(out) == cap_for(slot_type):
            break
    return tuple(out)


def coverage(lang: str) -> Dict[str, int]:
    """How many values each registered type yields for one language."""
    return {slot_type: len(values_for(slot_type, lang)) for slot_type in sorted(registry())}
