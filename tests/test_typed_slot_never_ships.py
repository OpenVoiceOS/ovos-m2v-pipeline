"""A typed slot never ships as a placeholder, on either side of the corpus.

The train side fills a typed slot from the parser for its type, and the
number that was supposed to prove it -- `train_rows_with_an_unfilled_slot`
-- cannot: `expand` strips the type, so an unfilled typed slot reaches a
finished row as `{offset}`, the same shape an UNTYPED slot leaves behind
legitimately under INTENT-1 5.4. 43,502 en-US rows of the published v6.1
corpus carry a brace on purpose, and a typed placeholder would sit inside
that figure unseen. So the type is read back per row from the row's own
template, which keeps the colon form.

The gold side cannot fill anything: a gold row is a sentence somebody says.
Filling a slot there would invent a phrasing nobody wrote, and keeping the
brace would score a brace. So gold refuses the row, and the two sides agree
on the one thing that matters: no brace of a typed slot ever reaches a row.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("ovos_spec_tools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_eval  # noqa: E402
import build_from_skills as bfs  # noqa: E402
import typed_slot_census  # noqa: E402


def row(utterance, template, lang="en-US"):
    return {"lang": lang, "label": "s:i", "utterance": utterance,
            "template": template}


def test_a_typed_slot_left_as_a_brace_is_found():
    hits = bfs.typed_slot_placeholders(
        [row("wake me in {offset}", "wake me in {number:offset}")])
    assert len(hits) == 1


def test_a_typed_slot_whose_type_was_never_stripped_is_found():
    hits = bfs.typed_slot_placeholders(
        [row("wake me in {number:offset}", "wake me in {number:offset}")])
    assert len(hits) == 1


def test_an_untyped_slot_left_as_a_brace_is_not_a_hit():
    """The control. INTENT-1 5.4 leaves an untyped slot literal on purpose,
    and a check that flags it flags 43,502 legitimate rows of v6.1."""
    assert bfs.typed_slot_placeholders([row("play {genre}", "play {genre}")]) == []


def test_a_filled_typed_slot_is_not_a_hit():
    assert bfs.typed_slot_placeholders(
        [row("wake me in five minutes", "wake me in {number:offset}")]) == []


def test_the_check_sees_one_bad_row_among_good_ones():
    rows = [row("play {genre}", "play {genre}"),
            row("wake me in 5", "wake me in {number:offset}"),
            row("wake me in {offset}", "wake me in {number:offset}")]
    hits = bfs.typed_slot_placeholders(rows)
    assert hits == ["en-US s:i: wake me in {offset}"]


@pytest.mark.parametrize("utterance", [
    "set an alarm for {offset}",
    "set an alarm for {number:offset}",
    "turn it (up|down)",
    "turn it [up]",
    "wake me | tomorrow",
])
def test_gold_template_syntax_is_refused(utterance):
    assert build_eval.TEMPLATE_SYNTAX.search(utterance)


@pytest.mark.parametrize("utterance", [
    "set an alarm for seven",
    "what is the weather like",
    "wake me in five minutes",
])
def test_a_real_gold_sentence_is_not_refused(utterance):
    """The control for the check above: it must let a sentence through."""
    assert not build_eval.TEMPLATE_SYNTAX.search(utterance)


def test_the_census_reports_in_and_out_per_locale():
    cells = typed_slot_census.in_out([
        {"lang": "en-US", "type": "number", "slot": "offset"},
        {"lang": "en-US", "type": "date", "slot": "when"},
    ])
    assert cells["en-US"]["in"] == 2
    assert cells["en-US"]["out"] == 2
    assert cells["en-US"]["dropped_types"] == {}


def test_a_type_with_no_generator_for_the_locale_is_out_and_named():
    """N in is not N out wherever a locale cannot fill a type, and the cell
    says which type, so the gap is never a mystery."""
    cells = typed_slot_census.in_out([
        {"lang": "en-US", "type": "timezone", "slot": "tz"},
        {"lang": "en-US", "type": "number", "slot": "offset"},
    ])
    assert cells["en-US"] == {"in": 2, "out": 1,
                              "dropped_types": {"timezone": 1}}


SETUP = '''
SKILL_NAME = "ovos-skill-demo"
PLUGIN_ENTRY_POINT = f"{SKILL_NAME}.openvoiceos=ovos_skill_demo:DemoSkill"
'''


def _gold_workspace(tmp_path, lines):
    import json
    import subprocess

    import yaml
    ws = tmp_path / "ws"
    root = ws / "ovos" / "ovos-skill-demo"
    (root / "test" / "end2end").mkdir(parents=True)
    (root / "setup.py").write_text(SETUP)
    (root / "test" / "end2end" / "golden_utterances.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines))
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
    return sources


def test_a_gold_row_with_a_typed_slot_is_refused_end_to_end(tmp_path):
    import json
    sources = _gold_workspace(tmp_path, [
        {"utterance": "wake me in {number:offset}", "intent_label": "wake"},
        {"utterance": "wake me in five minutes", "intent_label": "wake"},
    ])
    out = tmp_path / "eval"
    assert build_eval.main(["--sources", str(sources), "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["test_rows"] == 1
    assert manifest["gold_rows_refused_for_template_syntax"] == 1
    assert manifest["test_rows_with_template_syntax"] == 0
    census = json.loads((out / "census.json").read_text())
    cell = census["ovos-skill-demo"]["en-US"]
    assert cell["in"] == 2 and cell["out"] == 1 and cell["template_syntax"] == 1
    rows = [json.loads(ln) for ln
            in (out / "test.jsonl").read_text().splitlines() if ln]
    assert [r["utterance"] for r in rows] == ["wake me in five minutes"]


def test_the_build_refuses_a_typed_placeholder_that_reached_a_row(
        tmp_path, monkeypatch, capsys):
    """The bug this gate exists for: the type lookup comes back empty, the
    typed slot falls through to the untyped path, and the row ships a brace
    that the unfilled-slot count cannot distinguish from a legitimate one.
    """
    import json
    import subprocess

    import yaml
    ws = tmp_path / "ws"
    root = ws / "ovos" / "ovos-skill-demo"
    (root / "locale" / "en-US").mkdir(parents=True)
    (root / "setup.py").write_text(SETUP)
    (root / "locale" / "en-US" / "wake.intent").write_text(
        "wake me in {number:offset}\n")
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
    argv = ["build_from_skills.py", "--sources", str(sources),
            "--labels", str(labels), "--out", str(tmp_path / "out"),
            "--min-languages", "0", "--min-test-rows", "0",
            "--min-labels-scored", "0"]

    # The control: with the type lookup working, the slot is filled and the
    # build passes.
    monkeypatch.setattr(sys, "argv", list(argv))
    assert bfs.main() == 0, capsys.readouterr().err
    manifest = json.loads(
        (tmp_path / "out" / "manifest.json").read_text())
    assert manifest["train_rows_with_a_typed_slot_placeholder"] == 0
    assert manifest["train_rows"] > 0

    # The defect: the type lookup returns nothing, so nothing marks the slot
    # as typed and the brace survives into a row.
    monkeypatch.setattr(bfs, "declared_slot_types", lambda *_: {})
    monkeypatch.setattr(sys, "argv", list(argv))
    assert bfs.main() == 1
    assert "carry a typed slot as a placeholder" in capsys.readouterr().err
