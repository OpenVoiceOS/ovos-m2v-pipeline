"""``_entity_values`` must discard a malformed template's partial output.

``ovos_spec_tools.expansion.iter_expand`` is a generator that yields before
it validates: the template ``[maybe]`` yields ``"maybe"`` and only then
raises ``MalformedTemplate``. A caller that appends into its result list
while iterating keeps that partial yield even though the template as a
whole was rejected. ``_entity_values`` feeds the entity value set that
slot-filling draws from, so a partial keep silently widens every entity
that shares the file with a broken line.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

BUILD_DATASET_PATH = Path(__file__).resolve().parents[1] / "train" / "build_dataset.py"


@pytest.fixture
def build_dataset(monkeypatch):
    spec = importlib.util.spec_from_file_location("build_dataset", BUILD_DATASET_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_dataset"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("build_dataset", None)


def _values_for(build_dataset, monkeypatch, text: str):
    monkeypatch.setattr(build_dataset, "git_show", lambda repo, rev, path: text)
    return build_dataset._entity_values(Path("unused"), "unused", "unused")


def test_malformed_template_alone_contributes_no_values(build_dataset, monkeypatch):
    values = _values_for(build_dataset, monkeypatch, "[maybe]\n")
    assert values == []


def test_valid_template_beside_a_malformed_one_keeps_its_full_values(
    build_dataset, monkeypatch
):
    text = "(red|blue)\n[maybe]\n"
    values = _values_for(build_dataset, monkeypatch, text)
    assert sorted(values) == ["blue", "red"]


def test_a_fully_valid_template_is_unaffected(build_dataset, monkeypatch):
    values = _values_for(build_dataset, monkeypatch, "(red|blue)\n")
    assert sorted(values) == ["blue", "red"]
