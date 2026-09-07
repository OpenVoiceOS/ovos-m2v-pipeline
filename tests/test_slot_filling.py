"""`{slot}` placeholders must be filled the same way the runtime prototype
pipeline fills them (`ovos_m2v_pipeline.slots.expand_entities`), not exported
literally. No row may ever reach the corpus with a `{` in it.
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
    weather = git_repo(ws / "weather", {
        "pkg/locale/en-US/weather.intent":
            "what is the weather in {location}\n"
            "what is the weather in {location}\n"
            "how is the weather in {location}\n",
        "pkg/locale/en-US/location.entity":
            "paris\nlondon\ntokyo\n",
    })

    cfg = {
        "version": 2,
        "workspace": str(ws),
        "git_sources": [
            {"id": "weather", "kind": "plugin_intents", "family": "ocp",
             "pipeline_id": "ocp", "path": "weather", "revision": weather,
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


def test_no_row_contains_a_literal_brace(built):
    train = [json.loads(l) for l in (built / "train.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (built / "test.jsonl").read_text().splitlines()]
    for row in train + test:
        assert "{" not in row["utterance"], row


def test_slot_is_filled_with_a_registered_entity_value(built):
    train = [json.loads(l) for l in (built / "train.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (built / "test.jsonl").read_text().splitlines()]
    utterances = {r["utterance"] for r in train + test}
    assert any(city in u for u in utterances for city in ("paris", "london", "tokyo"))
