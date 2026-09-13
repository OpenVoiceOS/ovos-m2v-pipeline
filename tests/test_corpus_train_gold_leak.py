"""A gold sentence must never also appear as a training row.

The prior docstring argued the two sides cannot overlap because they are
read from different files -- locale resources for training,
``golden_utterances*.jsonl`` for gold. That is a claim about where a string
was read from, not about what the string is: a gold sentence and an
expansion of that skill's own template are both somebody reaching for the
obvious phrasing of the same intent, and they collide constantly. These
tests build a real skill repo with a colliding pair, run the actual builder
end to end, and check the written output -- not the design, the output.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ovos_spec_tools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_from_skills as bfs  # noqa: E402


def _write_skill_repo(root, files):
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@test.invalid", "-c", "user.name=test",
         "commit", "-q", "-m", "skill"],
        cwd=root, check=True)
    rev = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    return rev


def _run(tmp_path, key, files, monkeypatch, min_test_rows=0,
         min_labels_scored=0, no_gold=False):
    """Run the real builder end to end; return (rc, stderr, out_dir)."""
    workspace = tmp_path / "workspace"
    repo = workspace / key
    if no_gold:
        files = {k: v for k, v in files.items()
                if not k.startswith("golden_utterances")}
    rev = _write_skill_repo(repo, files)

    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump({
        "workspace": str(workspace),
        "skill_refs": {"refs": {key: rev}},
    }))

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "build_from_skills.py",
        "--sources", str(sources),
        "--out", str(out),
        "--min-labels", "0",
        "--min-languages", "0",
        "--min-test-rows", str(min_test_rows),
        "--min-labels-scored", str(min_labels_scored),
    ])
    return out


def _build(tmp_path, key, files, monkeypatch, capsys, **kwargs):
    """Run the real builder end to end and return (train, test, manifest)."""
    out = _run(tmp_path, key, files, monkeypatch, **kwargs)
    rc = bfs.main()
    assert rc == 0, capsys.readouterr().err

    train = [json.loads(l) for l in
             (out / "train.jsonl").read_text().splitlines() if l]
    test = [json.loads(l) for l in
            (out / "test.jsonl").read_text().splitlines() if l]
    manifest = json.loads((out / "manifest.json").read_text())
    return train, test, manifest


SKILL = "ovos-skill-fixture"
SKILL_FILES = {
    "locale/en-US/pick.intent": "play {genre}\n",
    "locale/en-US/genre.entity": "rock\njazz\n",
    "golden_utterances.jsonl": "\n".join([
        json.dumps({"utterance": "play rock", "intent_label": "pick"}),
    ]) + "\n",
}


def test_a_gold_utterance_that_also_trains_is_removed_from_train_only(
        tmp_path, monkeypatch, capsys):
    train, test, manifest = _build(
        tmp_path, SKILL, SKILL_FILES, monkeypatch, capsys)

    train_utterances = {r["utterance"] for r in train}
    test_utterances = {r["utterance"] for r in test}

    assert "play rock" not in train_utterances
    assert "play rock" in test_utterances
    # jazz never appeared in gold, and survives.
    assert "play jazz" in train_utterances
    assert manifest["train_gold_overlap_removed"] == 1
    assert manifest["train_gold_overlap_after_fix"] == 0


def test_case_and_whitespace_variants_of_a_gold_row_are_also_removed(
        tmp_path, monkeypatch, capsys):
    # The gold row carries different case and extra whitespace than the
    # training expansion it collides with -- the normalisation must fold
    # both before comparing, or this pair would slip through unremoved.
    files = dict(SKILL_FILES)
    files["golden_utterances.jsonl"] = json.dumps(
        {"utterance": "  PLAY   Rock  ", "intent_label": "pick"}) + "\n"
    train, test, manifest = _build(tmp_path, SKILL, files, monkeypatch, capsys)

    train_utterances = {r["utterance"] for r in train}
    assert "play rock" not in train_utterances
    assert "play jazz" in train_utterances
    assert manifest["train_gold_overlap_removed"] == 1
    assert manifest["train_gold_overlap_after_fix"] == 0


def test_an_accent_difference_is_not_removed(tmp_path, monkeypatch, capsys):
    files = {
        "locale/pt-PT/pick.intent": "toca {genre}\n",
        "locale/pt-PT/genre.entity": "musica\n",
        "golden_utterances.jsonl": json.dumps(
            {"utterance": "toca música", "intent_label": "pick"}) + "\n",
    }
    train, test, manifest = _build(tmp_path, SKILL, files, monkeypatch, capsys)

    train_utterances = {r["utterance"] for r in train}
    # "toca musica" (no accent) is a different string than the gold
    # "toca música" after lowercasing and whitespace collapse alone --
    # this normalisation is exact-match, not accent-folding, and must
    # leave the pair distinct.
    assert "toca musica" in train_utterances
    assert manifest["train_gold_overlap_removed"] == 0
    assert manifest["train_gold_overlap_after_fix"] == 0


def test_manifest_overlap_is_zero_and_silenced_templates_are_reported(
        tmp_path, monkeypatch, capsys):
    # "play rock" is the ONLY expansion of the first template, so removing
    # it silences that template entirely; "play jazz" survives from the
    # second entity value, so the template as a whole still keeps a row.
    files = {
        "locale/en-US/pick.intent": "play {genre}\n",
        "locale/en-US/genre.entity": "rock\n",
        "locale/en-US/queue.intent": "queue up {genre}\n",
        "locale/en-US/single.entity": "rock\n",
        "golden_utterances.jsonl": "\n".join([
            json.dumps({"utterance": "play rock", "intent_label": "pick"}),
            json.dumps({"utterance": "queue up rock",
                       "intent_label": "queue"}),
        ]) + "\n",
    }
    # Rename the entity file used by `queue.intent` so it fills from a
    # single value, guaranteeing that template's only row is the gold
    # collision and nothing else is left behind for it.
    del files["locale/en-US/single.entity"]
    files["locale/en-US/genre.entity"] = "rock\n"

    train, test, manifest = _build(tmp_path, SKILL, files, monkeypatch, capsys)

    assert manifest["train_gold_overlap_removed"] == 2
    assert manifest["train_gold_overlap_after_fix"] == 0
    assert manifest["templates_silenced"] > 0
    assert "play {genre}" in manifest["templates_silenced_detail"]
    assert "queue up {genre}" in manifest["templates_silenced_detail"]
    assert any(label.endswith(":pick")
               for label in manifest["labels_with_a_silenced_template"])
    assert any(label.endswith(":queue")
               for label in manifest["labels_with_a_silenced_template"])


def test_a_missing_gold_glob_fails_the_build_instead_of_scoring_nothing(
        tmp_path, monkeypatch, capsys):
    # A gold glob that resolves to nothing (a renamed or moved
    # golden_utterances*.jsonl) must not be able to build a corpus that
    # reports a clean overlap of zero by having nothing left to overlap
    # with: the test-side floors must catch it.
    out = _run(tmp_path, SKILL, SKILL_FILES, monkeypatch,
               min_test_rows=1, min_labels_scored=1, no_gold=True)
    rc = bfs.main()
    err = capsys.readouterr().err
    assert rc != 0
    assert "test rows" in err
    assert not (out / "train.jsonl").exists()


def test_a_gold_side_below_the_test_rows_floor_fails_and_names_the_floor(
        tmp_path, monkeypatch, capsys):
    out = _run(tmp_path, SKILL, SKILL_FILES, monkeypatch,
               min_test_rows=2, min_labels_scored=0)
    rc = bfs.main()
    err = capsys.readouterr().err
    assert rc != 0
    assert "floor 2" in err
    assert not (out / "train.jsonl").exists()


def test_a_gold_row_ending_in_punctuation_removes_its_unpunctuated_twin(
        tmp_path, monkeypatch, capsys):
    # The gold row carries a trailing "?" that its training twin never
    # had. ovos-utterance-normalizer strips exactly this kind of edge
    # punctuation before an intent engine ever compares the two, so at
    # match time they are one string; the removal must treat them as one
    # string too.
    files = dict(SKILL_FILES)
    files["golden_utterances.jsonl"] = json.dumps(
        {"utterance": "play rock?", "intent_label": "pick"}) + "\n"
    train, test, manifest = _build(tmp_path, SKILL, files, monkeypatch, capsys)

    train_utterances = {r["utterance"] for r in train}
    assert "play rock" not in train_utterances
    assert "play jazz" in train_utterances
    assert manifest["train_gold_overlap_removed"] == 1
    assert manifest["train_gold_overlap_after_fix"] == 0
    assert manifest["train_gold_overlap_after_fix_punct_insensitive"] == 0
