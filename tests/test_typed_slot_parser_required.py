"""A missing typed-slot parser fails the build; it is never an empty sample.

The defect: every generator imported its parser on its first line, and
`values_for` read the ImportError as "this language has no generator". An
absent package therefore emptied the type in every language at once, and
`fill` dropped every sentence carrying that slot. A date-time template that
yields 1,728 rows with `ovos-number-parser` installed yielded 0 without it,
and the build reported success.
"""
import collections
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ovos_spec_tools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_from_skills as bfs  # noqa: E402
import typed_slots as ts  # noqa: E402

PARSER_OF = {
    "number": "ovos_number_parser",
    "date": "ovos_date_parser",
    "duration": "ovos_date_parser",
    "color": "ovos_color_parser",
    "language": "ovos_lang_parser",
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    ts.values_for.cache_clear()
    yield
    ts.values_for.cache_clear()


@pytest.mark.parametrize("slot_type", sorted(PARSER_OF))
def test_a_missing_parser_raises_instead_of_returning_nothing(
        slot_type, monkeypatch):
    # The control: installed, the type has English values.
    assert ts.values_for(slot_type, "en-US")
    ts.values_for.cache_clear()
    monkeypatch.setitem(sys.modules, PARSER_OF[slot_type], None)
    with pytest.raises(RuntimeError, match=PARSER_OF[slot_type]):
        ts.values_for(slot_type, "en-US")


def test_a_missing_transformer_raises_instead_of_sampling_every_type(
        monkeypatch):
    """Without the plugin nobody knows which types the runtime binds."""
    monkeypatch.setitem(sys.modules, "ovos_typed_slots_transformer", None)
    with pytest.raises(RuntimeError, match="ovos_typed_slots_transformer"):
        ts.resolvable()


def test_fill_does_not_drop_a_typed_template_when_the_parser_is_missing(
        monkeypatch):
    """The builder-level shape of the loss, on the template that measured it."""
    template = ("what time (is it|will it be) (in|after) {number:offset} "
                "(hour|minute)[s] [from now]")
    assert bfs.fill(template, {}, {}, collections.Counter(), "en-US")
    ts.values_for.cache_clear()
    monkeypatch.setitem(sys.modules, "ovos_number_parser", None)
    with pytest.raises(RuntimeError, match="ovos_number_parser"):
        bfs.fill(template, {}, {}, collections.Counter(), "en-US")


SETUP = '''
SKILL_NAME = "ovos-skill-demo"
PLUGIN_ENTRY_POINT = f"{SKILL_NAME}.openvoiceos=ovos_skill_demo:DemoSkill"
'''


def _workspace(tmp_path, intents):
    ws = tmp_path / "ws"
    root = ws / "ovos" / "ovos-skill-demo"
    root.mkdir(parents=True)
    (root / "setup.py").write_text(SETUP)
    for (lang, name), body in intents.items():
        (root / "locale" / lang).mkdir(parents=True, exist_ok=True)
        (root / "locale" / lang / f"{name}.intent").write_text(body)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t.invalid", "-c", "user.name=t",
                    "commit", "-q", "-m", "skill"], cwd=root, check=True)
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         check=True).stdout.strip()
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump({
        "workspace": str(ws),
        "skill_refs": {"refs": {"ovos/ovos-skill-demo": sha}}}))
    labels = tmp_path / "labels.yaml"
    labels.write_text(yaml.safe_dump({
        "version": 1,
        "baseline": {"corpus": "fixture", "labels_trained": 0, "labels": []},
        "removed": [], "added": []}))
    return ["build_from_skills.py", "--sources", str(sources),
            "--labels", str(labels), "--out", str(tmp_path / "out"),
            "--min-languages", "0", "--min-test-rows", "0",
            "--min-labels-scored", "0"]


def test_the_build_refuses_a_typed_template_that_produced_no_row(
        tmp_path, monkeypatch, capsys):
    """The gate, independent of the raise: a sampler that comes back empty
    in every language is the missing-parser shape, whatever caused it."""
    argv = _workspace(tmp_path, {
        ("en-US", "wake"): "wake me in {number:offset} minutes\n",
        ("en-US", "hello"): "hello there\n"})
    monkeypatch.setattr(sys, "argv", list(argv))
    assert bfs.main() == 0, capsys.readouterr().err
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["train_rows"] == 7

    monkeypatch.setattr(bfs.typed_slots, "values_for", lambda *_: ())
    monkeypatch.setattr(sys, "argv", list(argv))
    assert bfs.main() == 1
    err = capsys.readouterr().err
    assert "typed-slot templates produced no row" in err
    assert "number" in err and "in any language" in err


def test_a_language_gap_is_reported_and_not_refused(
        tmp_path, monkeypatch, capsys):
    """The control: a type with no generator for ONE language drops that
    language's template by design, and the build still passes."""
    argv = _workspace(tmp_path, {
        ("en-US", "wake"): "wake me in {number:offset} minutes\n",
        ("fi-FI", "wake"): "herätä minut {number:offset} minuutin päästä\n"})
    real = ts.values_for
    monkeypatch.setattr(bfs.typed_slots, "values_for",
                        lambda t, lang: () if lang == "fi-FI" else real(t, lang))
    monkeypatch.setattr(sys, "argv", list(argv))
    assert bfs.main() == 0, capsys.readouterr().err
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["typed_slot_templates_with_no_row_language_gap"] == 1
    assert manifest["typed_slot_templates_with_no_row_unexplained"] == 0


def test_a_loss_inside_a_language_that_fills_the_type_is_refused(
        tmp_path, monkeypatch, capsys):
    """A template with no row where the same type filled another template in
    the same language lost its rows to something other than the language."""
    argv = _workspace(tmp_path, {
        ("en-US", "wake"): "wake me in {number:offset} minutes\n",
        ("en-US", "nap"): "let me nap for {number:span} hours\n"})
    real = bfs.fill
    monkeypatch.setattr(bfs, "fill", lambda t, *a: [] if "nap" in t
                        else real(t, *a))
    monkeypatch.setattr(sys, "argv", list(argv))
    assert bfs.main() == 1
    assert "let me nap" in capsys.readouterr().err


def test_one_empty_type_of_two_is_a_language_gap(
        tmp_path, monkeypatch, capsys):
    """`fill` drops a sentence when ANY of its typed slots has no values, so
    a two-type template whose second type is empty in its language is a gap,
    even though its first type fills other templates there. ovos_lang_parser
    has no fi-FI wordlist; number has fi-FI words and digits."""
    assert ts.values_for("language", "fi-FI") == ()
    assert ts.values_for("number", "fi-FI")
    argv = _workspace(tmp_path, {
        ("fi-FI", "say"): "sano {number:n} sanaa kielellä {language:l}\n",
        ("fi-FI", "wake"): "herätä minut {number:n} minuutin päästä\n",
        ("en-US", "say"): "say {number:n} words in {language:l}\n"})
    monkeypatch.setattr(sys, "argv", list(argv))
    assert bfs.main() == 0, capsys.readouterr().err
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["typed_slot_templates_with_no_row_language_gap"] == 1
    assert manifest["typed_slot_templates_with_no_row_unexplained"] == 0


def test_a_malformed_typed_template_is_not_blamed_on_a_parser(
        tmp_path, monkeypatch, capsys):
    """An unclosed group is counted as `template_would_not_expand`; it says
    nothing about whether the number parser works, so this gate leaves it
    out instead of refusing the build for a missing parser."""
    argv = _workspace(tmp_path, {
        ("en-US", "wake"): "wake me in {number:n} minutes\n",
        ("en-US", "nap"): "let me nap for {number:n} (hours\n"})
    monkeypatch.setattr(sys, "argv", list(argv))
    assert bfs.main() == 0, capsys.readouterr().err
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["stats"]["template_would_not_expand"] == 1
    assert manifest["typed_slot_templates_with_no_row_unexplained"] == 0
