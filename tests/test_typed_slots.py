"""A typed slot must leave the corpus as a surface, never as a brace.

The defect these tests exist for is a training row that reads
"what time will it be in {number:offset} minutes". The classifier then learns
a brace, and the runtime never says one.
"""
import re

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import typed_slots as ts  # noqa: E402

PLACEHOLDER = re.compile(r"\{[a-z_]+:[a-z_]+\}")


def test_the_registry_is_read_and_not_declared():
    """The list of types is the specification's, not this module's."""
    from ovos_spec_tools import REGISTERED_TYPES
    assert ts.registry() == frozenset(REGISTERED_TYPES)


def test_the_resolvable_set_is_read_from_the_runtime_plugin():
    from ovos_typed_slots_transformer import TypedSlotsTransformer
    assert ts.resolvable() == frozenset(TypedSlotsTransformer.supported_types)


def test_a_type_the_runtime_cannot_bind_is_never_sampled():
    """The rule that keeps train and runtime honest.

    A row whose value the runtime can never bind is worse than a row that is
    missing: it teaches a surface nothing will ever resolve.
    """
    for slot_type in ts.unresolvable():
        assert ts.values_for(slot_type, "en-US") == ()


@pytest.mark.parametrize("lang", ["en-US", "de-DE", "ca-ES", "fr-FR", "kab"])
def test_number_values_are_that_language_and_not_english(lang):
    values = ts.values_for("number", lang)
    assert values, f"no number surfaces for {lang}"
    if lang != "en-US":
        assert set(values) != set(ts.values_for("number", "en-US")), (
            f"{lang} numbers are identical to en-US, which is not {lang} data")


@pytest.mark.parametrize("lang", ["en-US", "de-DE", "ca-ES", "kab"])
@pytest.mark.parametrize("slot_type", ["number", "date", "duration"])
def test_no_sample_carries_template_syntax(lang, slot_type):
    """A value that carries a brace re-introduces the defect one layer down."""
    for value in ts.values_for(slot_type, lang):
        assert "{" not in value and "}" not in value, value
        assert not PLACEHOLDER.search(value), value


def test_a_language_with_no_generator_yields_nothing_rather_than_english():
    """The control: silence is a real answer, an English word is not."""
    values = ts.values_for("number", "xx-XX")
    assert values == () or set(values) != set(ts.values_for("number", "en-US"))


def test_values_are_distinct():
    assert len(set(ts.values_for("number", "en-US"))) == len(ts.values_for("number", "en-US"))


def test_the_sample_is_capped():
    """The cap counts surfaces, so a two-surface type keeps its values."""
    for slot_type in ts.resolvable():
        assert len(ts.values_for(slot_type, "en-US")) <= ts.cap_for(slot_type)


def test_a_one_surface_type_is_still_capped_at_the_sample_count():
    """The control: the new cap must not loosen every other type.

    `date` has one surface per value, so its cap does not move. If `cap_for`
    ever multiplied the wrong types, this is what fails.
    """
    assert ts.cap_for("date") == ts.SAMPLES_PER_SLOT
    assert ts.cap_for("number") == ts.SAMPLES_PER_SLOT * 2
    assert len(ts.values_for("date", "en-US")) <= ts.SAMPLES_PER_SLOT


def test_the_number_sample_carries_both_surfaces():
    """Digits for what an ASR sends, words for what a person says."""
    values = ts.values_for("number", "en-US")
    assert [v for v in values if v.isdigit()], values
    assert [v for v in values if not v.isdigit()], values


def test_every_anchor_keeps_both_of_its_surfaces():
    """The contract the corpus depends on, asserted per anchor.

    A cap that counted values rather than surfaces would silently drop whole
    anchors from the end of the list, and a test that only asks whether SOME
    digit and SOME word survive would still pass. This asks per anchor.
    """
    values = ts.values_for("number", "en-US")
    for anchor in ts._NUMBERS:
        assert str(anchor) in values, (anchor, values)
    assert len([v for v in values if not v.isdigit()]) == len(ts._NUMBERS), values


def test_a_language_with_no_wordlist_still_gets_the_digits():
    """The digits are language-neutral, so no language is left with no rows.

    Before the ruling a language whose wordlist the parser lacks produced no
    numeric surface at all, and every template naming a number slot was
    dropped for it.
    """
    assert [v for v in ts.values_for("number", "xx-XX") if v.isdigit()]


def test_coverage_reports_every_registered_type():
    coverage = ts.coverage("en-US")
    assert set(coverage) == ts.registry()
    for slot_type in ts.unresolvable():
        assert coverage[slot_type] == 0


def test_every_resolvable_type_produces_english_values():
    """The positive control for the whole module: a type the runtime binds
    and this module cannot sample is a row the corpus silently lacks. The
    set is read from the plugin, so a new member fails here the day it
    lands (`language` did, with ovos-typed-slots-transformer 0.0.1a4)."""
    for slot_type in sorted(ts.resolvable()):
        assert ts.values_for(slot_type, "en-US"), slot_type


@pytest.mark.parametrize("lang", ["en-US", "de-DE", "ca-ES", "fr-FR", "kab"])
def test_language_values_are_that_language_and_not_english(lang):
    values = ts.values_for("language", lang)
    assert values, f"no language surfaces for {lang}"
    if lang != "en-US":
        assert set(values) != set(ts.values_for("language", "en-US")), (
            f"{lang} language names are identical to en-US, which is not {lang} data")


def test_language_names_are_the_parsers_own():
    """The value is what `ovos_lang_parser` would recognise back, not a
    hand-written translation: the sampler and the runtime read one wordlist."""
    from ovos_lang_parser import pronounce_lang
    assert ts.values_for("language", "ca-ES") == tuple(
        pronounce_lang(code, lang="ca-ES") for code in ("en", "fr", "de"))


def test_a_language_the_lang_parser_lacks_yields_nothing():
    """fi-FI has no bundled wordlist; the answer is silence, never English."""
    assert ts.values_for("language", "fi-FI") == ()
