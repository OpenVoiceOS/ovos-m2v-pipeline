"""`filters.drop_non_alpha` must be Unicode-aware and independent of the
utterance column's pandas dtype.

`_ALPHA_RE` matched fine against a plain `object`-dtype Series, but pandas
3.x's default string storage promotes the utterance column to its own
`str` dtype as soon as it is built from a list of Python strings, and
`Series.str.contains(_ALPHA_RE, ...)` then silently stops matching
non-Latin scripts -- Cyrillic, Greek, CJK -- zeroing out every such row,
even though the exact same text survives fine as plain `object` dtype. The
fix applies `str.isalpha()` per plain Python string via `.map()`, bypassing
pandas' own regex engine entirely.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[1] / "train" / "build_dataset.py"

NON_LATIN = {
    "uk-UA": "зупинити зараз",
    "ru-RU": "остановить воспроизведение",
    "bg-BG": "спри музиката сега",
    "el-GR": "σταμάτα τη μουσική",
    "ja-JP": "音楽を止めて",
    "kab": "ḥbes tura",
}


SETUP_PY = '''
URL = "https://github.com/OpenVoiceOS/ovos-skill-stop-fixture"
AUTHOR = "OpenVoiceOS"
PYPI_NAME = URL.split("/")[-1]
SKILL_ID = f"{PYPI_NAME.lower()}.{AUTHOR.lower()}"
SKILL_PKG = PYPI_NAME.lower().replace('-', '_')
SKILL_CLAZZ = "StopFixtureSkill"
PLUGIN_ENTRY_POINT = f"{SKILL_ID}={SKILL_PKG}:{SKILL_CLAZZ}"
'''


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
    skill = git_repo(ws / "skills" / "stop-fixture", {
        "setup.py": SETUP_PY,
        "ovos_skill_stop_fixture/locale/en-US/stop.intent": "stop\nstop it\n",
    })
    # ovos-localize's JSONL export shape: one line per (lang, skill, intent,
    # utterance). Two lines per locale so the label clears
    # `min_rows_per_label` -- the point under test is the content filter,
    # not the rare-label one.
    lines = "\n".join(
        json.dumps({"lang": lang, "skill": "ovos-skill-stop-fixture",
                   "intent": "stop", "text": text})
        for lang, text in NON_LATIN.items()
        for text in (text, text + " !")
    )
    localize = git_repo(ws / "localize", {
        "data/datasets/classification/all.jsonl": lines + "\n",
    })

    cfg = {
        "version": 2,
        "workspace": str(ws),
        "git_sources": [
            {"id": "localize", "kind": "localize_jsonl", "path": "localize",
             "revision": localize,
             "files": "data/datasets/classification/*.jsonl",
             "columns": {"lang": "lang", "skill_id": "skill", "intent": "intent",
                        "utterance": "text"},
             "exclude": {"file_type": ["voc"]}},
        ],
        "hf_sources": [],
        "golden": {
            "policy": "stratified", "provenance_column": "source",
            "shared": {"path": "golden.jsonl", "default_lang": "en-US"},
            "per_skill": {"files": "test/end2end/golden_utterances*.jsonl",
                          "lang_from_filename": True, "default_lang": "en-US"},
            "exclude": {"needs_manual": True},
        },
        "skill_refs": {"refs": {"skills/stop-fixture": skill}},
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


def test_non_latin_locales_survive_the_alpha_filter(built):
    train = [json.loads(l) for l in (built / "train.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (built / "test.jsonl").read_text().splitlines()]
    langs = {r["lang"] for r in train + test}

    missing = set(NON_LATIN) - langs
    assert not missing, f"non-Latin locales dropped by the alpha filter: {missing}"

    n_rows = sum(1 for r in train + test if r["lang"] in NON_LATIN)
    assert n_rows == 2 * len(NON_LATIN), (
        f"expected {2 * len(NON_LATIN)} non-Latin rows (2 per locale), got {n_rows}")
