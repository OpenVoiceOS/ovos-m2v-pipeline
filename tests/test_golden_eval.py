"""The golden evaluation scores a model on each pinned skill's own gold files.

A fixture skill repo carries gold rows that must each land in one bucket: a
scored row, a needs_manual row, a dialog-only row, a row with no utterance
text, a renamed label, a row whose text is a training row, and a label the
model never trained. A stub predictor stands in for the model, so the test
checks the selection and the arithmetic, not a model download.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ovos_spec_tools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import golden_eval  # noqa: E402

SKILL = "ovos-skill-fixture.openvoiceos"


def git_repo(path: Path, files: dict) -> str:
    path.mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for name, text in files.items():
        f = path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(path), *args], check=True, env=env,
                       capture_output=True)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def jsonl(rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


class StubModel:
    """Predicts `greet` for anything with 'hello' or 'hallo', `bye` otherwise."""
    classes_ = [f"{SKILL}:greet", f"{SKILL}:bye", f"{SKILL}:time_until"]

    def predict(self, utterances):
        return [f"{SKILL}:greet" if ("hello" in u or "hallo" in u) else f"{SKILL}:bye"
                for u in utterances]


@pytest.fixture
def run(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    rev = git_repo(ws / "skills" / "ovos-skill-fixture", {
        "test/end2end/golden_utterances.jsonl": jsonl([
            {"skill_id": SKILL, "utterance": "hello there", "intent_label": "greet.intent"},
            {"skill_id": SKILL, "utterance": "hello friend", "intent_label": "greet.intent"},
            {"skill_id": SKILL, "utterance": "see you later", "intent_label": "bye.intent"},
            # predicted wrong by the stub: counts as a scored, wrong row
            {"skill_id": SKILL, "utterance": "hello and goodbye", "intent_label": "bye.intent"},
            {"skill_id": SKILL, "utterance": "maybe later", "intent_label": "bye.intent",
             "needs_manual": True},
            {"skill_id": SKILL, "utterance": "tell me a joke"},
            # a row with a label and no utterance text: there is nothing to
            # predict, and the count must say so rather than drop the row
            {"skill_id": SKILL, "utterance": "   ", "intent_label": "greet.intent"},
            # the training rows contain this text: refused, never scored
            {"skill_id": SKILL, "utterance": "Goodbye!", "intent_label": "bye.intent"},
            # an old label renamed in RENAMED_LABELS below; "countdown" does
            # not fold to "time_until", so only the rename table resolves it
            {"skill_id": SKILL, "utterance": "how long until friday", "intent_label": "countdown"},
            # a label the model never trained
            {"skill_id": SKILL, "utterance": "sing a song", "intent_label": "sing.intent"},
        ]),
        "test/end2end/golden_utterances_de-DE.jsonl": jsonl([
            {"skill_id": SKILL, "utterance": "hallo zusammen", "intent_label": "greet.intent"},
            {"skill_id": SKILL, "utterance": "hallo und tschüss", "intent_label": "bye.intent"},
        ]),
    })
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump({"workspace": str(ws),
                                       "skill_refs": {"refs": {"skills/ovos-skill-fixture": rev}}}))
    train = tmp_path / "train.jsonl"
    train.write_text(jsonl([{"utterance": "goodbye", "label": f"{SKILL}:bye", "lang": "en-US"},
                            {"utterance": "hi", "label": f"{SKILL}:greet", "lang": "en-US"}]))
    monkeypatch.setitem(golden_eval.RENAMED_LABELS, f"{SKILL}:countdown", f"{SKILL}:time_until")
    out = tmp_path / "out"
    code = golden_eval.main(["--sources", str(sources), "--train", str(train),
                             "--out", str(out), "--model", str(tmp_path / "model")],
                            predictor=StubModel())
    return code, json.loads((out / "golden_eval.json").read_text()), out


def test_every_row_lands_in_exactly_one_bucket(run):
    code, report, _ = run
    assert code == 0
    assert report["gold_rows_read"] == 12
    assert report["excluded"] == {"needs_manual": 1, "no_intent_label": 1, "no_utterance": 1,
                                  "unparsable": 0, "unresolved_label": 1,
                                  "refused_train_overlap": 1, "gold_duplicates": 0}
    assert report["rows_scored"] == 7
    assert 12 == report["rows_scored"] + sum(report["excluded"].values())


def test_a_gold_row_that_is_a_training_row_is_refused(run):
    _, report, _ = run
    assert [r["utterance"] for r in report["refused_train_overlap_rows"]] == ["Goodbye!"]
    # positive control: a bye row whose text is not in train is scored
    assert report["per_label"][f"{SKILL}:bye"]["rows"] == 3


def test_a_renamed_label_is_scored_under_its_new_name(run):
    _, report, _ = run
    assert report["per_label"][f"{SKILL}:time_until"]["rows"] == 1
    assert report["unresolved_labels"] == {f"{SKILL}:sing": 1}


def test_accuracy_per_language_and_label(run):
    _, report, out = run
    # en-US: hello there, hello friend, see you later right; hello and goodbye
    # wrong (stub says greet); how long until friday wrong (stub says bye)
    assert report["per_language"]["en-US"] == {"rows": 5, "correct": 3, "accuracy": 0.6}
    assert report["per_language"]["de-DE"] == {"rows": 2, "correct": 1, "accuracy": 0.5}
    assert report["correct"] == 4 and report["accuracy"] == round(4 / 7, 4)
    assert report["per_label"][f"{SKILL}:bye"]["most_frequent_wrong"] == {
        "label": f"{SKILL}:greet", "rows": 2}
    assert "| de-DE | 2 | 1 | 0.5 |" in (out / "golden_eval.md").read_text()


def test_no_scored_rows_fails_the_floor(tmp_path):
    ws = tmp_path / "ws"
    rev = git_repo(ws / "skills" / "ovos-skill-fixture", {"README.md": "no gold\n"})
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump({"workspace": str(ws),
                                       "skill_refs": {"refs": {"skills/ovos-skill-fixture": rev}}}))
    train = tmp_path / "train.jsonl"
    train.write_text(jsonl([{"utterance": "hi"}]))
    code = golden_eval.main(["--sources", str(sources), "--train", str(train),
                             "--out", str(tmp_path / "out"), "--model", "m"],
                            predictor=StubModel())
    assert code == 1
