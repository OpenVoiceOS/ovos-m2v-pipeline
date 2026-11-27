"""A corpus label whose skill id is not in `skill_refs`/`skill_id_aliases`
must be reported as a loud per-repo WARNING with its row count, not just
folded into `unresolved_labels` where a dry run has to be read to notice it.
"""
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


def test_unresolved_registry_gap_warns_loudly_with_row_count(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    pytest.importorskip("yaml")

    ws = tmp_path / "ws"
    tracker = git_repo(ws / "tracker", {"skills/intents_en.csv": (
        "domain,intent,utterance\n"
        "ovos-common-reading-pipeline-plugin,ReadContent,read me a story\n"
        "ovos-common-reading-pipeline-plugin,ReadContent,read this book\n")})

    cfg = {
        "version": 2,
        "workspace": str(ws),
        "git_sources": [
            {"id": "tracker", "kind": "tracker_csv", "path": "tracker",
             "revision": tracker, "files": "skills/intents_{lang}.csv",
             "langs": ["en"]},
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
        # deliberately no alias for ovos-common-reading-pipeline-plugin:
        # its rows must be reported as a registry gap, not resolved.
        "skill_id_aliases": {},
        "skill_blacklist": [],
        "intent_blacklist": [],
        "filters": {"min_chars": 2, "max_chars": 160, "min_words": 1,
                    "drop_bare_slot": True, "drop_non_alpha": True,
                    "min_rows_per_label": 1},
        "split": {"test_size": 0.5, "seed": 42, "stratify": "label"},
    }
    (ws / "golden.jsonl").write_text("", encoding="utf-8")
    import yaml
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    r = subprocess.run([sys.executable, str(BUILDER), "--sources", str(sources),
                        "--workspace", str(ws), "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING" in r.stderr
    assert "ovos-common-reading-pipeline-plugin" in r.stderr
    assert "2" in r.stderr
