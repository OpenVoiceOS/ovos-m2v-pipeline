"""A language whose entire corpus turns out to be unfillable `{slot}`
templates must not just vanish -- `exclusions.json` names it, its row
counts, and the exact (skill id, slot) that caused the drop, so a missing
language is a worklist item (add an `.entity` file, or rework the
template) instead of an unexplained hole in the record.
"""
import json
import re
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


# fr-FR fixture row naming two distinct unfillable slots
FR_ROUTE_TEMPLATE = "trajet de {origin} vers {dest}"

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
        # fr-FR: one row names two distinct unfillable slots -- it must be
        # attributed to exactly one reason bucket, not counted once per slot.
        "pkg/locale/fr-FR/route.intent": FR_ROUTE_TEMPLATE + "\n",
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


def test_exclusion_manifest_arithmetic_closes_for_every_language(built):
    """`{location}` has two registered values, so en-US's two raw template
    rows explode into four filled rows -- more rows kept than raw rows
    before the fill. `rows_before` must be counted at the same
    (post-explode) granularity as `rows_kept`, or `rows_dropped` goes
    negative here, which is worse than not reporting a count at all.
    """
    excl = json.loads((built / "exclusions.json").read_text())
    for lang, e in excl["languages"].items():
        assert e["rows_before"] == e["rows_kept"] + e["rows_dropped"], lang
        assert e["rows_dropped"] >= 0, lang
        assert (sum(r["rows"] for r in e["unfilled_slot_reasons"])
                == e["rows_dropped_unfilled_slot"]), lang

    en = excl["languages"]["en-US"]
    assert en["rows_before"] == 4
    assert en["rows_kept"] == 4
    assert en["rows_dropped"] == 0


def test_two_slot_row_attributed_once_not_per_slot(built):
    """One dropped row naming both `{origin}` and `{dest}` must count once
    towards `rows_dropped_unfilled_slot`, and its reasons must sum to that
    same one, not two.
    """
    excl = json.loads((built / "exclusions.json").read_text())
    fr = excl["languages"]["fr-FR"]
    assert fr["rows_dropped_unfilled_slot"] == 1
    assert sum(r["rows"] for r in fr["unfilled_slot_reasons"]) == 1
    assert len(fr["unfilled_slot_reasons"]) == 1
    # the one bucket names BOTH slots: fixing only `dest` would drop the
    # same row again on `origin`, and the manifest must say so up front
    fixture_slots = sorted(re.findall(r"\{(\w+)\}", FR_ROUTE_TEMPLATE))
    assert fr["unfilled_slot_reasons"][0]["slot"] == "+".join(fixture_slots)
