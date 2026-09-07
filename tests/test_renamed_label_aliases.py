"""Corpus rows filed under a pre-rename intent spelling must reach the label
the pinned skill registers today.

The HF-hosted corpora (the ovos-localize export, the lang-support tracker
CSVs, the GitLocalize export) still carry the spellings the default skills
used before the Adapt-to-`.intent` migration and the weather/alerts/volume
unification wave. Without an alias entry those rows resolve to nothing and
are dropped as `unresolved_labels`.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[1] / "train" / "build_dataset.py"

#: (repo, skill id the corpus files rows under, old corpus intent spelling,
#: skill id the repo's entry point declares, intent the repo registers).
RENAMES = [
    ("ovos-skill-mark1-ctrl", "ovos-skill-mark1-ctrl.openvoiceos",
     "EnclosureEyesBlink", "ovos-skill-mark1-ctrl.openvoiceos", "blink"),
    ("ovos-skill-weather", "ovos-skill-weather.openvoiceos",
     "hourly_forecast", "ovos-skill-weather.openvoiceos", "weather"),
    ("ovos-skill-weather", "ovos-skill-weather.openvoiceos",
     "is_rain", "ovos-skill-weather.openvoiceos", "weather_condition"),
    ("ovos-skill-volume", "ovos-skill-volume.openvoiceos",
     "volume.high", "ovos-skill-volume.openvoiceos", "volume_level"),
    ("ovos-skill-confucius-quotes", "ovos-skill-confucius-quotes.openvoiceos",
     "ConfuciusBirth", "ovos-skill-confucius-quotes.openvoiceos",
     "confucius_lifespan"),
    # the wallpapers corpus files rows under the repo name; the entry point
    # declares the words in the other order
    ("ovos-skill-wallpapers", "ovos-skill-wallpapers.openvoiceos",
     "MakeWallpaperIntent", "skill-ovos-wallpapers.openvoiceos",
     "make_wallpaper"),
]


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


def test_pre_rename_corpus_rows_resolve_to_the_registered_intent(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    yaml = pytest.importorskip("yaml")
    import json

    ws = tmp_path / "ws"
    by_repo = {}
    for i, (repo, corpus_id, old, skill_id, new) in enumerate(RENAMES):
        files = by_repo.setdefault(repo, {"pyproject.toml":
            f'[project.entry-points."ovos.plugin.skill"]\n'
            f'"{skill_id}" = "s:S"\n'})
        files[f"locale/en-US/{new}.intent"] = f"fixture utterance {i}\n"
        # a never-renamed intent, so the build still produces a splittable
        # dataset when the renamed rows are dropped
        files["locale/en-US/keeper.intent"] = "keeper fixture line\n"
    refs = {repo: git_repo(ws / repo, files) for repo, files in by_repo.items()}
    # two rows each so every label clears the stratification floor
    rows = [f"{corpus_id},{old},corpus utterance {i} {w}"
            for i, (_r, corpus_id, old, _s, _n) in enumerate(RENAMES)
            for w in ("alpha", "beta")]
    rows += [f"{corpus_id},keeper,keeper utterance {i} {w}"
             for i, (_r, corpus_id, *_x) in enumerate(RENAMES)
             for w in ("alpha", "beta")]

    tracker = git_repo(ws / "tracker", {"skills/intents_en.csv":
                       "domain,intent,utterance\n" + "\n".join(rows) + "\n"})
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
        "skill_refs": {"refs": refs},
        "skill_id_aliases": {},
        "skill_blacklist": [],
        "intent_blacklist": [],
        "filters": {"min_chars": 2, "max_chars": 160, "min_words": 1,
                    "drop_bare_slot": True, "drop_non_alpha": True,
                    "min_rows_per_label": 2},
        "split": {"test_size": 0.5, "seed": 42, "stratify": "label"},
    }
    (ws / "golden.jsonl").write_text("", encoding="utf-8")
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(BUILDER), "--sources", str(sources),
                        "--workspace", str(ws), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["unresolved_labels"]["detail"] == {}

    import pandas as pd
    df = pd.concat([pd.read_parquet(out / "train.parquet"),
                    pd.read_parquet(out / "test.parquet")])
    counts = df["label"].value_counts().to_dict()
    for _repo, corpus_id, old, skill_id, new in RENAMES:
        assert f"{corpus_id}:{old}" not in counts
        # the two aliased corpus rows, plus the one line the skill's own
        # `locale/en-US/<new>.intent` fixture ships
        assert counts[f"{skill_id}:{new}"] == 3, counts
