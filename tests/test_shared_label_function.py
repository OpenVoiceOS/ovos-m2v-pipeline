"""One label function on both sides of the corpus.

A train row and a gold row for one intent must carry one label, and that
label's skill id is the id the repository's entry point declares, not the
repository's name. ``ovos-skill-easter-eggs`` declares
``skill-easter-eggs.openvoiceos``; before ``train/skill_labels.py`` the train
side wrote ``ovos-skill-easter-eggs.openvoiceos`` and the gold side copied the
row's own field, so the two sides never met and the model learnt a label no
skill answers to."""
import collections
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ovos_spec_tools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

import build_eval  # noqa: E402
import build_from_skills as bfs  # noqa: E402
import skill_labels  # noqa: E402

SETUP = '''
from setuptools import setup
SKILL_NAME = "skill-easter-eggs"
SKILL_PKG = SKILL_NAME.replace("-", "_")
PLUGIN_ENTRY_POINT = f"{SKILL_NAME}.openvoiceos={SKILL_PKG}:EasterEggsSkill"
setup(name=SKILL_NAME, entry_points={"ovos.plugin.skill": PLUGIN_ENTRY_POINT})
'''


def _repo(root, files):
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t.invalid", "-c", "user.name=t",
                    "commit", "-q", "-m", "skill"], cwd=root, check=True)
    return root


def _easter_eggs(tmp_path):
    return _repo(tmp_path / "ovos-skill-easter-eggs", {
        "setup.py": SETUP,
        "locale/en-US/intents/hal_intent.intent": "open the pod bay doors\n",
        "test/end2end/golden_utterances.jsonl":
            '{"skill_id": "skill-easter-eggs.openvoiceos", "utterance": "hal open the doors", '
            '"intent_label": "hal_intent.intent", "lang": "en-US"}\n',
    })


def test_the_train_side_labels_with_the_declared_skill_id(tmp_path):
    repo = _easter_eggs(tmp_path)
    stats = collections.Counter()
    templates, _, _ = bfs.read_skill(repo, "HEAD", "ovos-skill-easter-eggs", stats, set())
    assert {label for _, label, _ in templates} == {"skill-easter-eggs.openvoiceos:hal_intent"}
    assert stats["skill_id_assumed_from_repo_name"] == 0


def test_the_gold_side_labels_with_the_same_function(tmp_path):
    repo = _easter_eggs(tmp_path)
    stats = collections.Counter()
    rows = build_eval.read_gold(repo, "HEAD", "ovos-skill-easter-eggs", stats)
    assert [r["label"] for r in rows] == ["skill-easter-eggs.openvoiceos:hal_intent"]


def test_both_sides_agree_on_one_repo(tmp_path):
    repo = _easter_eggs(tmp_path)
    stats = collections.Counter()
    templates, _, _ = bfs.read_skill(repo, "HEAD", "ovos-skill-easter-eggs", stats, set())
    train_labels = {label for _, label, _ in templates}
    gold_labels = {r["label"] for r in
                   build_eval.read_gold(repo, "HEAD", "ovos-skill-easter-eggs", stats)}
    assert gold_labels <= train_labels


def test_a_gold_row_field_does_not_override_the_declared_id(tmp_path):
    repo = _repo(tmp_path / "ovos-skill-easter-eggs", {
        "setup.py": SETUP,
        "test/end2end/golden_utterances.jsonl":
            '{"skill_id": "somebody-else.openvoiceos", "utterance": "hal open the doors", '
            '"intent_label": "hal_intent"}\n',
    })
    rows = build_eval.read_gold(repo, "HEAD", "ovos-skill-easter-eggs", collections.Counter())
    assert rows[0]["label"] == "skill-easter-eggs.openvoiceos:hal_intent"


def test_a_repo_without_packaging_is_assumed_and_counted(tmp_path):
    repo = _repo(tmp_path / "ovos-skill-bare", {
        "locale/en-US/intents/hi.intent": "hi there\n"})
    stats = collections.Counter()
    templates, _, _ = bfs.read_skill(repo, "HEAD", "ovos-skill-bare", stats, set())
    assert {label for _, label, _ in templates} == {"ovos-skill-bare.openvoiceos:hi"}
    assert stats["skill_id_assumed_from_repo_name"] == 1


def test_no_spelling_fallback_on_the_gold_side(tmp_path):
    """`count_to_N` in a gold file stays `count_to_N`: the census refuses it,
    the reader does not fold it onto `count_to_n`."""
    repo = _repo(tmp_path / "ovos-skill-count", {
        "setup.py": SETUP.replace("skill-easter-eggs", "ovos-skill-count"),
        "test/end2end/golden_utterances.jsonl":
            '{"utterance": "count to ten", "intent_label": "count_to_N.intent"}\n',
    })
    rows = build_eval.read_gold(repo, "HEAD", "ovos-skill-count", collections.Counter())
    assert rows[0]["label"] == "ovos-skill-count.openvoiceos:count_to_N"
    assert skill_labels.label("x", "a.intent") == "x:a"
