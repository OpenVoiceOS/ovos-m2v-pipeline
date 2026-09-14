"""An inline vocabulary reference `<name>` (OVOS-INTENT-1 3.7) must expand from
the `.voc` file of the same locale, never ship as literal text. A sentence
whose reference has no `.voc` in its locale is dropped and counted in
`manifest.json` under `dropped_unresolved_voc_rows`.
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
    plugin = git_repo(ws / "plugin", {
        # en-US: two references that resolve, one of them nested in a group
        # and through a second vocabulary
        "pkg/locale/en-US/hi.intent":
            "say <greeting> now\n"
            "(please say|say) <greeting> to me\n",
        "pkg/locale/en-US/greeting.voc": "hello\n<short_greeting>\n",
        "pkg/locale/en-US/short_greeting.voc": "hi\n",
        # en-US: a reference with no .voc anywhere -> 1 sentence dropped
        "pkg/locale/en-US/bye.intent": "wave <farewell> now\nsay goodbye\nsay bye now\n",
        # pt-PT: greeting.voc exists only under en-US -> 1 sentence dropped
        "pkg/locale/pt-PT/hi.intent": "diz <greeting> agora\ndiz ola\ndiz oi agora\n",
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


def test_a_reference_cycle_is_malformed_not_a_crash():
    """OVOS-INTENT-1 4.1 step 1: a reference cycle is malformed (3.6). It must
    raise the KeyError the reader counts and drops, never recurse without end."""
    pytest.importorskip("pandas")
    pytest.importorskip("yaml")
    sys.path.insert(0, str(BUILDER.parent))
    import build_dataset  # noqa: E402

    vocabs = {"a": ["x <b>"], "b": ["y <a>"]}
    with pytest.raises(KeyError):
        build_dataset._resolve_voc(["say <a>"], vocabs)
    # positive control: the same helper resolves a chain without a cycle
    assert build_dataset._resolve_voc(["say <a>"], {"a": ["x <b>"], "b": ["y"]}) == ["say x y"]


def _rows(out):
    return [json.loads(l) for name in ("train.jsonl", "test.jsonl")
            for l in (out / name).read_text().splitlines()]


def test_no_row_contains_an_inline_vocabulary_reference(built):
    rows = _rows(built)
    assert rows
    for row in rows:
        assert "<" not in row["utterance"] and ">" not in row["utterance"], row


def test_reference_expands_to_every_member_of_its_locale_vocabulary(built):
    utterances = {r["utterance"] for r in _rows(built) if r["lang"].startswith("en")}
    assert {"say hello now", "say hi now", "say hello to me", "please say hi to me"} <= utterances


def test_unresolved_reference_is_dropped_and_counted(built):
    utterances = {r["utterance"] for r in _rows(built)}
    # positive control: the resolvable lines of the same two files are kept
    assert {"say goodbye", "diz ola"} <= utterances
    assert not any(u.startswith("wave") for u in utterances)
    assert not any(u.startswith("diz") and u not in {"diz ola", "diz oi agora"}
                   for u in utterances)
    manifest = json.loads((built / "manifest.json").read_text())
    assert manifest["dropped_unresolved_voc_rows"] == 2
