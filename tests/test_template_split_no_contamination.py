"""The train/test split must be by TEMPLATE, not by row: every expansion of
one `.intent` template line lands entirely in one split. A row-level
stratified split lets near-identical expansions of the same line leak across
train and test, inflating held-out accuracy.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[1] / "train" / "build_dataset.py"


def git_repo(path: Path, files: dict) -> str:
    path.mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True,
                                    env=env, capture_output=True)
    run("init", "-q", "-b", "main")
    for name, text in files.items():
        f = path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "fixture")
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def built(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    pytest.importorskip("yaml")

    ws = tmp_path / "ws"
    # one template line per label, each expanding via (alt|alt) into several
    # near-identical rows -- these must never straddle the split.
    lines = "\n".join(
        f"(play|start|put on) the {n}(st|nd|rd|th) song" for n in range(2, 12))
    plugin = git_repo(ws / "plugin", {
        "pkg/locale/en-US/play.intent": lines + "\n",
    })

    cfg = {
        "version": 2,
        "workspace": str(ws),
        "git_sources": [
            {"id": "plugin", "kind": "plugin_intents", "family": "ocp",
             "pipeline_id": "ocp", "path": "plugin", "revision": plugin,
             "files": "**/locale/*/*.intent"},
        ],
        "hf_sources": [],
        "golden": {
            "policy": "stratified", "provenance_column": "source",
            "shared": {"path": "golden.jsonl", "default_lang": "en-US"},
            "per_skill": {"files": "test/end2end/golden_utterances*.jsonl",
                          "lang_from_filename": True, "default_lang": "en-US"},
            "exclude": {"needs_manual": True},
        },
        "skill_refs": {"refs": {}},
        "skill_id_aliases": {},
        "skill_blacklist": [],
        "intent_blacklist": [],
        "filters": {"min_chars": 2, "max_chars": 160, "min_words": 1,
                    "drop_bare_slot": True, "drop_non_alpha": True,
                    "min_rows_per_label": 2},
        "split": {"test_size": 0.5, "seed": 42, "stratify": "label"},
    }
    (ws / "golden.jsonl").write_text("", encoding="utf-8")
    import yaml
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(BUILDER), "--sources", str(sources),
                        "--workspace", str(ws), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


def _template_of(utterance: str) -> str:
    """The (alt|alt) branch doesn't matter, only the numeral does, so group
    by the digit each expansion carries -- every row from one template line
    shares one digit and must land on one side of the split."""
    for tok in utterance.split():
        if tok[:-2].isdigit() or tok[:-3].isdigit():
            return tok
    return utterance


def test_no_template_straddles_train_and_test(built):
    train = {json.loads(l)["utterance"]
            for l in (built / "train.jsonl").read_text().splitlines()}
    test = {json.loads(l)["utterance"]
           for l in (built / "test.jsonl").read_text().splitlines()}
    train_templates = {_template_of(u) for u in train}
    test_templates = {_template_of(u) for u in test}
    straddling = train_templates & test_templates
    assert not straddling, f"templates on both sides of the split: {straddling}"
