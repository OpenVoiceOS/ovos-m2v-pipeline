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
    for slot_type in ts.resolvable():
        assert len(ts.values_for(slot_type, "en-US")) <= ts.SAMPLES_PER_SLOT


def test_coverage_reports_every_registered_type():
    coverage = ts.coverage("en-US")
    assert set(coverage) == ts.registry()
    for slot_type in ts.unresolvable():
        assert coverage[slot_type] == 0


def test_the_four_resolvable_types_all_produce_english_values():
    """The positive control for the whole module."""
    for slot_type in sorted(ts.resolvable()):
        assert ts.values_for(slot_type, "en-US"), slot_type
