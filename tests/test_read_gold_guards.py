"""build_eval.read_gold: the order of the gold guards, and a row whose
utterance is not a string.

Every gold row that read_gold drops lands in exactly one counter. The order of
the guards decides which one, so a row that trips two of them is the only test
that holds the order in place: a blank unlabelled row counts as a row without
an utterance, not as a row that asserts a dialog, and a needs_manual blank row
counts as needs_manual. A row whose utterance is a number used to reach
``.strip()`` and raise AttributeError, which ended the whole build."""
import collections
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ovos_spec_tools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

import build_eval  # noqa: E402


def _gold_repo(root, rows):
    path = root / "test" / "end2end" / "golden_utterances.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t.invalid", "-c", "user.name=t",
                    "commit", "-q", "-m", "skill"], cwd=root, check=True)
    return root


def _read(tmp_path, rows):
    repo = _gold_repo(tmp_path / "ovos-skill-demo", rows)
    stats = collections.Counter()
    census = {}
    out = build_eval.read_gold(repo, "HEAD", "ovos-skill-demo", stats, census)
    return out, stats, census["ovos-skill-demo"]["en-US"]


def test_a_blank_unlabelled_row_counts_as_a_row_without_an_utterance(tmp_path):
    rows, stats, cell = _read(tmp_path, [
        {"utterance": "", "intent_label": None, "lang": "en-US"}])
    assert rows == []
    assert stats["gold_without_utterance"] == 1
    assert stats["gold_asserts_a_dialog_not_an_intent"] == 0
    assert cell["in"] == 1
    assert cell["no_utterance"] == 1
    assert "no_intent" not in cell


def test_a_needs_manual_blank_row_counts_as_needs_manual(tmp_path):
    rows, stats, cell = _read(tmp_path, [
        {"utterance": "   ", "intent_label": None, "lang": "en-US",
         "needs_manual": True}])
    assert rows == []
    assert stats["gold_needs_manual"] == 1
    assert stats["gold_without_utterance"] == 0
    assert stats["gold_asserts_a_dialog_not_an_intent"] == 0
    assert cell["needs_manual"] == 1
    assert "no_utterance" not in cell


def test_a_non_string_utterance_is_counted_and_does_not_raise(tmp_path):
    rows, stats, cell = _read(tmp_path, [
        {"utterance": 5, "intent_label": "hello.intent", "lang": "en-US"},
        {"utterance": "hello there", "intent_label": "hello.intent",
         "lang": "en-US"}])
    assert [r["utterance"] for r in rows] == ["hello there"]
    assert stats["gold_utterance_not_a_string"] == 1
    assert cell["in"] == 2
    assert cell["utterance_not_a_string"] == 1
    assert cell["out"] == 1
