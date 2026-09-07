"""A language whose entire corpus turns out to be unfillable `{slot}`
templates must not just vanish -- `exclusions.json` names it, its row
counts, and the exact (skill id, slot) that caused the drop, so a missing
language is a worklist item (add an `.entity` file, or rework the
template) instead of an unexplained hole in the record.
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
    # en-US: {location} is filled from the pinned .entity file, survives.
    # de-DE: {query} has no registered entity anywhere -- every one of its
    # rows is unfillable, so the language must end at zero rows.
    plugin = git_repo(ws / "plugin", {
        "pkg/locale/en-US/weather.intent":
            "what is the weather in {location}\nhow is the weather in {location}\n",
        "pkg/locale/en-US/location.entity": "paris\nlondon\n",
        "pkg/locale/de-DE/search.intent":
            "suche nach {query}\nfinde {query}\n",
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


def test_de_de_absent_from_output_frame(built):
    train = [json.loads(l) for l in (built / "train.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (built / "test.jsonl").read_text().splitlines()]
    langs = {r["lang"] for r in train + test}
    assert "de-DE" not in langs
    assert "en-US" in langs


def test_exclusions_json_records_the_absent_language(built):
    excl = json.loads((built / "exclusions.json").read_text())

    assert "de-DE" in excl["languages"]
    de = excl["languages"]["de-DE"]
    assert de["rows_before"] == 2
    assert de["rows_kept"] == 0
    assert de["rows_dropped"] == 2
    assert de["rows_dropped_unfilled_slot"] == 2
    reasons = de["unfilled_slot_reasons"]
    assert len(reasons) == 1
    assert reasons[0]["skill_id"] == "ocp"
    assert reasons[0]["slot"] == "query"
    assert reasons[0]["rows"] == 2

    assert "de-DE" in excl["languages_absent_from_output"]
    assert "en-US" not in excl["languages_absent_from_output"]

    en = excl["languages"]["en-US"]
    assert en["rows_kept"] > 0
    assert en["rows_dropped_unfilled_slot"] == 0
