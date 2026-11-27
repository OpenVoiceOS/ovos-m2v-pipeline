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


def test_bulk_entity_oversampling_does_not_flood_warnings(tmp_path):
    """`train/build_dataset.py` fills one `{slot}` template per corpus row,
    each an independent `expand_entities` call (see the comment above
    `_fill` in the builder). A single large `.entity` file therefore used
    to emit one WARNING per row from `slots.expand_entities` -- thousands
    of identical-shaped lines that buried anything else logged during the
    build. The per-row line is DEBUG; the builder logs one bounded summary
    instead, and a run must never regress back to a WARNING per row.
    """
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    pytest.importorskip("yaml")

    ws = tmp_path / "ws"
    import string
    alphabet = string.ascii_lowercase
    # `collect_entities` itself caps each entity's bucket at
    # MAX_ENTITY_EXPANSIONS, so a single-slot template can never overflow
    # the bound there. Two capped 2000-value entities in one template still
    # multiply to 2000*2000 combinations -- the real-world shape (two
    # `.entity`-backed slots in a template) this test reproduces at a
    # smaller size: 2000 places x 2 units is already 4000 > 2000.
    places = "\n".join(
        f"place{alphabet[i // 676 % 26]}{alphabet[i // 26 % 26]}{alphabet[i % 26]}"
        for i in range(2100))
    weather = git_repo(ws / "weather", {
        "pkg/locale/en-US/weather.intent":
            "what is the weather in {location} in {unit}\n"
            "how is the weather in {location} in {unit}\n",
        "pkg/locale/en-US/location.entity": places + "\n",
        "pkg/locale/en-US/unit.entity": "celsius\nfahrenheit\n",
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

    combined = r.stdout + r.stderr
    lines = combined.splitlines()
    # the per-row line must not reach the build output at all: a return to
    # one WARNING per corpus row shows up here as "expands to" lines.
    per_row = [l for l in lines if "expands to" in l]
    assert per_row == [], per_row
    assert any("[slots]" in l and "template fill(s)" in l
               for l in lines if "WARNING" in l), lines


def test_runtime_expand_entities_warns_and_records_nothing(monkeypatch):
    """The runtime pipeline calls `expand_entities` once per registration
    for the life of the process. It keeps the WARNING, because nothing there
    reads `oversample_stats()`, and it adds nothing to that list. The builder's
    opt-in call is the positive control: the same template is recorded."""
    from ovos_m2v_pipeline import slots

    # the OVOS logger does not propagate, so pytest's caplog sees nothing;
    # record the calls on the logger itself.
    calls = []
    monkeypatch.setattr(slots.LOG, "warning", lambda m, *a, **k: calls.append(("WARNING", m)))
    monkeypatch.setattr(slots.LOG, "debug", lambda m, *a, **k: calls.append(("DEBUG", m)))

    tmpl = ["go to {a} {b}"]
    ents = {"a": [f"a{i}" for i in range(60)], "b": [f"b{i}" for i in range(60)]}

    slots.reset_oversample_stats()
    for _ in range(5):
        out = slots.expand_entities(tmpl, ents)
    assert len(out) == slots.MAX_ENTITY_EXPANSIONS
    assert slots.oversample_stats() == []
    warned = [m for lvl, m in calls if lvl == "WARNING" and "expands to" in m]
    assert len(warned) == 5, calls

    slots.expand_entities(tmpl, ents, record_oversample=True)
    assert slots.oversample_stats() == [("go to {a} {b}", 3600)]
    slots.reset_oversample_stats()
