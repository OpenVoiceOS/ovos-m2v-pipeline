"""``{slot}`` placeholders carry the owning skill's own declared vocabulary
when ``--fill-slots`` is passed, instead of surviving into training rows as
literal placeholder text.

Real declared vocabulary means a same-named ``.entity`` file the skill
itself ships - never a hand-picked stand-in, and never a ``.voc`` file
(``.voc`` backs ``voc_match``/``<name>`` inline matching, not a captured
slot) - so a slot with no matching ``.entity`` keeps the pre-existing
literal-placeholder behaviour, and is flagged as likely mis-modeled when a
same-named ``.voc`` exists instead.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_dataset  # noqa: E402

BUILDER = Path(__file__).resolve().parents[1] / "train" / "build_dataset.py"


def test_fill_slot_placeholders_uses_declared_entity_vocab():
    df = pd.DataFrame([
        {"label": "weather.openvoiceos:weather_condition",
         "skill_id": "weather.openvoiceos", "lang": "en-US",
         "utterance": "is it {condition} today", "source": "tracker"},
    ])
    entity_vocab = {"weather.openvoiceos": {"en-US": {"condition": ["rainy", "foggy", "snowy"]}}}

    out, n_filled, flagged, unmatched = build_dataset.fill_slot_placeholders(
        df, entity_vocab, {}, cap=3, seed=1)

    assert flagged == []
    assert unmatched == []
    assert n_filled == len(out)
    assert 1 <= len(out) <= 3
    for sentence in out["utterance"]:
        assert "{condition}" not in sentence
        assert any(v in sentence for v in ("rainy", "foggy", "snowy"))


def test_fill_slot_placeholders_is_deterministic():
    df = pd.DataFrame([
        {"label": "weather.openvoiceos:weather_condition",
         "skill_id": "weather.openvoiceos", "lang": "en-US",
         "utterance": "is it {condition} today", "source": "tracker"},
    ])
    entity_vocab = {"weather.openvoiceos": {"en-US": {"condition": ["rainy", "foggy", "snowy", "windy"]}}}

    out_a, *_ = build_dataset.fill_slot_placeholders(df, entity_vocab, {}, cap=3, seed=7)
    out_b, *_ = build_dataset.fill_slot_placeholders(df, entity_vocab, {}, cap=3, seed=7)

    assert list(out_a["utterance"]) == list(out_b["utterance"])


def test_unmatched_slot_falls_back_to_literal_placeholder():
    """A slot with no declared vocabulary anywhere for the skill is left
    exactly as it is today - it is never invented."""
    df = pd.DataFrame([
        {"label": "acme.jarbasai:demo",
         "skill_id": "acme.jarbasai", "lang": "en-US",
         "utterance": "add {item} to the list", "source": "tracker"},
    ])

    out, n_filled, flagged, unmatched = build_dataset.fill_slot_placeholders(
        df, {}, {}, cap=3, seed=1)

    assert n_filled == 0
    assert flagged == []
    assert unmatched == ["acme.jarbasai:item"]
    assert list(out["utterance"]) == ["add {item} to the list"]


def test_voc_only_slot_is_flagged_not_filled():
    """A ``{slot}`` backed only by a same-named ``.voc`` (no ``.entity``) is
    a closed keyword set mis-modeled as a captured slot - it must never be
    filled from the ``.voc``, only flagged for the skill to fix."""
    df = pd.DataFrame([
        {"label": "weather.openvoiceos:weather_condition",
         "skill_id": "weather.openvoiceos", "lang": "en-US",
         "utterance": "is it {condition} today", "source": "tracker"},
    ])
    voc_stems = {"weather.openvoiceos": {"en-US": {"condition"}}}

    out, n_filled, flagged, unmatched = build_dataset.fill_slot_placeholders(
        df, {}, voc_stems, cap=3, seed=1)

    assert n_filled == 0
    assert unmatched == []
    assert flagged == ["weather.openvoiceos:condition"]
    assert list(out["utterance"]) == ["is it {condition} today"]


def test_expand_template_resolves_inline_voc_reference():
    vocab = {"eyes": ["googly eyes", "normal eyes"]}
    out = build_dataset.expand_template("blink <eyes>", vocab=vocab)
    assert out == ["blink googly eyes", "blink normal eyes"]


def test_expand_template_leaves_undefined_voc_reference_literal():
    out = build_dataset.expand_template("blink <eyes>", vocab={})
    assert out == ["blink <eyes>"]


def _git_repo(path: Path, files: dict) -> str:
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
URL = "https://github.com/OpenVoiceOS/ovos-skill-weather"
AUTHOR = "OpenVoiceOS"
PYPI_NAME = URL.split("/")[-1]
SKILL_ID = f"{PYPI_NAME.lower()}.{AUTHOR.lower()}"
SKILL_PKG = PYPI_NAME.lower().replace('-', '_')
SKILL_CLAZZ = "WeatherSkill"
PLUGIN_ENTRY_POINT = f"{SKILL_ID}={SKILL_PKG}:{SKILL_CLAZZ}"
'''


@pytest.fixture
def workspace_cfg(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    pytest.importorskip("yaml")

    ws = tmp_path / "ws"
    skill = _git_repo(ws / "skills" / "weather", {
        "setup.py": SETUP_PY,
        "ovos_skill_weather/locale/en-US/weather_condition.intent":
            "is it {condition} today\nwill it {condition} tomorrow\n",
        "ovos_skill_weather/locale/en-US/condition.entity":
            "rain\nsnow\nfog\n",
    })
    tracker = _git_repo(ws / "tracker", {"skills/intents_en.csv": (
        "domain,intent,utterance\n"
        "ovos-skill-weather,weather_condition.intent,is it {condition} today\n"
        "ovos-skill-weather,weather_condition.intent,will it {condition} tomorrow\n"
        "ovos-skill-weather,weather_condition.intent,is it rainy today\n"
        "ovos-skill-weather,weather_condition.intent,will it snow tomorrow\n")})

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
        "skill_refs": {"refs": {"skills/weather": skill}},
        "skill_id_aliases": {},
        "skill_blacklist": [],
        "intent_blacklist": [],
        "filters": {"min_chars": 2, "max_chars": 160, "min_words": 1,
                    "drop_bare_slot": True, "drop_non_alpha": True,
                    "min_rows_per_label": 2},
        "split": {"test_size": 0.5, "seed": 42, "stratify": "label"},
    }
    golden = ws / "golden.jsonl"
    golden.write_text("", encoding="utf-8")

    import yaml
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return ws, sources


def _run(ws, sources, out, *extra):
    r = subprocess.run([sys.executable, str(BUILDER), "--sources", str(sources),
                        "--workspace", str(ws), "--out", str(out), *extra],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _all_utterances(out):
    utterances = []
    for name in ("train.jsonl", "test.jsonl"):
        for line in (out / name).read_text(encoding="utf-8").splitlines():
            utterances.append(json.loads(line)["utterance"])
    return utterances


def test_default_run_keeps_literal_slot_placeholder(workspace_cfg, tmp_path):
    ws, sources = workspace_cfg
    out = tmp_path / "out_default"
    _run(ws, sources, out)
    utterances = _all_utterances(out)
    assert any("{condition}" in u for u in utterances)


def test_fill_slots_flag_replaces_placeholder_with_declared_entity_value(workspace_cfg, tmp_path):
    ws, sources = workspace_cfg
    out = tmp_path / "out_filled"
    _run(ws, sources, out, "--fill-slots")
    utterances = _all_utterances(out)
    assert not any("{condition}" in u for u in utterances)
    assert any(v in u for u in utterances for v in ("rain", "snow", "fog"))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["fill_slots_enabled"] is True
    assert manifest["fill_slots_rows_added"] > 0


def test_plugin_intents_expand_inline_voc_reference(tmp_path):
    """A ``plugin_intents`` source's ``<name>`` inline reference resolves
    against a sibling ``.voc`` file in the same locale directory, so the
    build produces real values rather than the literal ``<eyes>`` token."""
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    pytest.importorskip("yaml")

    ws = tmp_path / "ws"
    plugin = _git_repo(ws / "plugin", {
        "demo_plugin/locale/en-US/blink.intent": "blink <eyes>\n",
        "demo_plugin/locale/en-US/eyes.voc": "googly eyes\nnormal eyes\n",
    })

    cfg = {
        "version": 2,
        "workspace": str(ws),
        "git_sources": [
            {"id": "demo", "kind": "plugin_intents", "family": "demo",
             "pipeline_id": "demo", "path": "plugin", "revision": plugin,
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
    _run(ws, sources, out)
    utterances = _all_utterances(out)
    assert not any("<eyes>" in u for u in utterances)
    assert any(v in u for u in utterances for v in ("googly eyes", "normal eyes"))
