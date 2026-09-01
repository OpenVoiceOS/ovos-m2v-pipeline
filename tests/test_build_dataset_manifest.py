"""The manifest shape the plugin loads is a contract, so it gets a test.

The builder is driven against a throwaway workspace of tiny git repos rather
than the real corpus: the pinned revisions are a moving target and the real
build takes minutes, but the shape of `labels.json` must not drift.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[1] / "train" / "build_dataset.py"


def git_repo(path: Path, files: dict) -> str:
    """Create a repo holding *files* and return its commit sha."""
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


SETUP_PY = '''
URL = "https://github.com/OpenVoiceOS/ovos-skill-hello-world"
AUTHOR = "OpenVoiceOS"
PYPI_NAME = URL.split("/")[-1]
SKILL_ID = f"{PYPI_NAME.lower()}.{AUTHOR.lower()}"
SKILL_PKG = PYPI_NAME.lower().replace('-', '_')
SKILL_CLAZZ = "HelloWorldSkill"
PLUGIN_ENTRY_POINT = f"{SKILL_ID}={SKILL_PKG}:{SKILL_CLAZZ}"
'''


@pytest.fixture
def built(tmp_path):
    """Run the builder over a fixture workspace; return the output directory."""
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    pytest.importorskip("yaml")

    ws = tmp_path / "ws"

    ocp = git_repo(ws / "ocp", {
        "ocp_pipeline/locale/en-US/play.intent":
            "play music\nplay some jazz\nput on a record\nstart the music\n"})
    core = git_repo(ws / "core", {
        "ovos_core/intent_services/locale/en-us/stop.intent":
            "stop\nstop it\nhalt\ncancel that\n"})
    cq = git_repo(ws / "cq", {"README.md": "no intent resources of its own\n"})
    skill = git_repo(ws / "skills" / "hello", {
        "setup.py": SETUP_PY,
        "ovos_skill_hello_world/locale/en-US/hello.world.intent":
            "hello world\nhey world\ngreetings world\nhi world\n"})

    tracker = git_repo(ws / "tracker", {"skills/intents_en.csv": (
        "domain,intent,utterance\n"
        "common_query,common_query,how tall is the eiffel tower\n"
        "common_query,common_query,who wrote dune\n"
        "common_query,common_query,what is the capital of peru\n"
        "common_query,common_query,when did the wall fall\n"
        # a bare skill id, as the tracker actually spells it, and the
        # resource-file spelling of the intent: both must resolve
        "ovos-skill-hello-world,hello.world.intent,say hi to the world\n"
        "ovos-skill-hello-world,hello.world.intent,greet the world\n"
        "ovos-skill-hello-world,hello.world.intent,world greeting please\n"
        "ovos-skill-hello-world,hello.world.intent,hello there world\n")})

    golden = ws / "golden.jsonl"
    golden.write_text(json.dumps(
        {"skill_id": "ovos-skill-hello-world.openvoiceos",
         "intent_label": "hello.world", "utterance": "say hello world",
         "lang": "en-US"}) + "\n", encoding="utf-8")

    cfg = {
        "version": 2,
        "workspace": str(ws),
        "git_sources": [
            {"id": "tracker", "kind": "tracker_csv", "path": "tracker",
             "revision": tracker, "files": "skills/intents_{lang}.csv",
             "langs": ["en"]},
            {"id": "ocp", "kind": "plugin_intents", "family": "ocp",
             "pipeline_id": "ocp", "path": "ocp", "revision": ocp,
             "files": "**/locale/*/*.intent"},
            {"id": "cq", "kind": "plugin_intents", "family": "common_query",
             "pipeline_id": "common_query", "path": "cq", "revision": cq,
             "files": "**/locale/*/*.intent",
             "extra_intents": ["common_query"]},
            {"id": "stop", "kind": "plugin_intents", "family": "stop",
             "pipeline_id": "stop", "path": "core", "revision": core,
             "files": "ovos_core/intent_services/locale/*/*.intent"},
        ],
        "hf_sources": [],
        "golden": {
            "policy": "stratified", "provenance_column": "source",
            "shared": {"path": "golden.jsonl", "default_lang": "en-US"},
            "per_skill": {"files": "test/end2end/golden_utterances*.jsonl",
                          "lang_from_filename": True, "default_lang": "en-US"},
            "exclude": {"needs_manual": True},
        },
        "skill_refs": {"refs": {"skills/hello": skill}},
        "skill_id_aliases": {},
        "skill_blacklist": [],
        "intent_blacklist": [],
        "filters": {"min_chars": 2, "max_chars": 160, "min_words": 1,
                    "drop_bare_slot": True, "drop_non_alpha": True,
                    "min_rows_per_label": 2},
        "split": {"test_size": 0.5, "seed": 42, "stratify": "label"},
    }
    import yaml
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(BUILDER), "--sources", str(sources),
                        "--workspace", str(ws), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


def test_labels_json_carries_a_family_for_every_valid_label(built):
    manifest = json.loads((built / "labels.json").read_text())
    valid = manifest["valid_labels"]
    families = manifest["families"]

    assert valid, "the fixture corpus produced no labels"
    assert sorted(families) == sorted(valid)
    assert set(families.values()) <= {"skill", "ocp", "common_query", "stop",
                                      "persona"}


def test_special_labels_are_raw_not_bus_topics(built):
    """The plugin maps these to bus topics at match time, so the model ships
    them exactly as trained."""
    manifest = json.loads((built / "labels.json").read_text())
    valid = set(manifest["valid_labels"])

    for label, family in (("ocp:play", "ocp"), ("stop:stop", "stop"),
                          ("common_query:common_query", "common_query")):
        assert label in valid
        assert manifest["families"][label] == family

    assert not any("." in lbl.split(":", 1)[0] and lbl.startswith("ovos.")
                   for lbl in valid)


def test_skill_labels_are_registered_and_unsuffixed(built):
    manifest = json.loads((built / "labels.json").read_text())
    valid = set(manifest["valid_labels"])

    assert "ovos-skill-hello-world.openvoiceos:hello.world" in valid
    assert manifest["families"][
        "ovos-skill-hello-world.openvoiceos:hello.world"] == "skill"
    assert not any(lbl.endswith(".intent") for lbl in valid)


def test_manifest_records_the_pinned_revisions(built):
    report = json.loads((built / "manifest.json").read_text())
    assert report["labels"] == len(
        json.loads((built / "labels.json").read_text())["valid_labels"])
    assert report["ambiguous_residual_groups"] == 0
    assert set(report["outputs"]) == {"train.parquet", "train.jsonl",
                                      "test.parquet", "test.jsonl",
                                      "labels.json"}
