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
"""
from __future__ import annotations

import datetime
import functools
from typing import Dict, FrozenSet, List

SAMPLES_PER_SLOT = 3

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

    Falls back to the registry when the plugin is absent, so a build host
    without it samples everything rather than nothing, and the census then
    reports what could not be checked.
    """
    try:
        from ovos_typed_slots_transformer import TypedSlotsTransformer
    except ImportError:
        return registry()
    return frozenset(TypedSlotsTransformer.supported_types)


def unresolvable() -> FrozenSet[str]:
    """Registered types the runtime does not bind. Reported, never sampled."""
    return registry() - resolvable()


def _numbers(lang: str) -> List[str]:
    from ovos_number_parser import pronounce_number
    out = []
    for value in _NUMBERS:
        try:
            spoken = pronounce_number(value, lang=lang)
        except Exception:
            continue
        if spoken and "{" not in spoken:
            out.append(str(spoken))
    return out


def _dates(lang: str) -> List[str]:
    from ovos_date_parser import nice_date
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
    from ovos_date_parser import nice_duration
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
    try:
        from ovos_color_parser import lookup_name
        from ovos_color_parser.models import sRGBAColor
    except ImportError:
        return []
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
    try:
        from ovos_lang_parser import pronounce_lang
    except ImportError:
        return []
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
def values_for(slot_type: str, lang: str) -> tuple:
    """Up to SAMPLES_PER_SLOT surface values for one type in one language.

    An empty result is a real answer: this language has no generator for
    this type, and the caller must leave the template out rather than fill
    it with another language's words.
    """
    if slot_type not in resolvable():
        return ()
    generator = _GENERATORS.get(slot_type)
    if generator is None:
        return ()
    try:
        produced = generator(lang)
    except Exception:
        return ()
    seen, out = set(), []
    for value in produced:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
        if len(out) == SAMPLES_PER_SLOT:
            break
    return tuple(out)


def coverage(lang: str) -> Dict[str, int]:
    """How many values each registered type yields for one language."""
    return {slot_type: len(values_for(slot_type, lang)) for slot_type in sorted(registry())}
