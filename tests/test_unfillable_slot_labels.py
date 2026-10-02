"""A label with no test rows is thin for one of two reasons, and the fix
differs: a genuinely thin label wants a phrasing or a translation, and a
label whose every surviving phrasing in some language names a slot the
pinned refs register no entity values for wants an entity file instead. The
builder must tell the two apart rather than reporting both under
`labels_without_test_rows`.
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
    fixture = git_repo(ws / "fixture", {
        # splittable, so the split is never asked to stratify an all-thin
        # corpus: two templates, both fillable, two groups.
        "pkg/locale/en-US/weather.intent":
            "what is the weather in {location}\n"
            "how is the weather in {location}\n",
        "pkg/locale/en-US/location.entity":
            "paris\nlondon\ntokyo\n",
        # genuinely thin: one template, alternation-expanded to two rows,
        # one group -- stays whole in train, no slot involved.
        "pkg/locale/en-US/thin.intent":
            "turn (on|up) the light\n",
        # limited by an unfilled slot: one surviving template (no slot, two
        # rows, one group) plus one template whose slot has no entity file
        # anywhere in the fixture, so every one of its rows is dropped at
        # the fill step.
        "pkg/locale/en-US/limited.intent":
            "remind me to call (back|later)\n"
            "set a timer for {duration}\n",
    })

    cfg = {
        "version": 2,
        "workspace": str(ws),
        "git_sources": [
            {"id": "fixture", "kind": "plugin_intents", "family": "ocp",
             "pipeline_id": "ocp", "path": "fixture", "revision": fixture,
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


def _manifest(built):
    return json.loads((built / "manifest.json").read_text())


def test_unfillable_slot_label_is_not_reported_as_thin(built):
    manifest = _manifest(built)
    thin = set(manifest["labels_without_test_rows"]["names"])
    assert "ocp:limited" not in thin
    assert "ocp:thin" in thin


def test_unfillable_slot_label_names_its_slot_and_language(built):
    manifest = _manifest(built)
    limited = manifest["labels_limited_by_unfilled_slots"]
    assert limited["labels"] == 1
    detail = limited["detail"]["ocp:limited"]
    assert detail["slots"] == ["duration"]
    assert detail["languages"] == ["en-US"]
    assert "duration" in detail["reason"]
    assert "en-US" in detail["reason"]


def test_thin_label_carries_no_slot_reason(built):
    manifest = _manifest(built)
    assert "ocp:thin" not in manifest["labels_limited_by_unfilled_slots"]["detail"]
