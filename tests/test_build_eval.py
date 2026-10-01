"""build_eval: gold rows at the pins, one file per locale, a census, a gate."""
import collections
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ovos_spec_tools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

import build_eval  # noqa: E402

SETUP = '''
SKILL_NAME = "ovos-skill-demo"
PLUGIN_ENTRY_POINT = f"{SKILL_NAME}.openvoiceos=ovos_skill_demo:DemoSkill"
'''


def _repo(root, files):
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t.invalid", "-c", "user.name=t",
                    "commit", "-q", "-m", "skill"], cwd=root, check=True)
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _workspace(tmp_path):
    ws = tmp_path / "ws"
    sha = _repo(ws / "ovos" / "ovos-skill-demo", {
        "setup.py": SETUP,
        "test/end2end/golden_utterances.jsonl": "\n".join([
            json.dumps({"utterance": "hello there", "intent_label": "hello.intent"}),
            json.dumps({"utterance": "hello there", "intent_label": "hello.intent"}),
            json.dumps({"utterance": "skip me", "intent_label": "hello", "needs_manual": True}),
            json.dumps({"utterance": "fallback speaks", "intent_label": None}),
            "not json",
        ]) + "\n",
        "test/end2end/golden_utterances_pt-PT.jsonl":
            json.dumps({"utterance": "olá", "intent_label": "hello"}) + "\n",
    })
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump({
        "workspace": str(ws), "skill_refs": {"refs": {"ovos/ovos-skill-demo": sha}}}))
    return sources


def test_a_skill_with_no_golden_file_is_in_the_census(tmp_path):
    """T-3011: a cloned skill that ships no golden file must appear in
    census.json with a reason, or a reader cannot tell it from a skill the
    build never looked at (review of #219: 21 of 64 skill_refs absent)."""
    sources = _workspace(tmp_path)
    ws = tmp_path / "ws"
    sha = _repo(ws / "ovos" / "ovos-skill-nogold", {"setup.py": SETUP.replace("demo", "nogold"),
                                                      "README.md": "no golden file\n"})
    cfg = yaml.safe_load(sources.read_text())
    cfg["skill_refs"]["refs"]["ovos/ovos-skill-nogold"] = sha
    sources.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "eval"
    assert build_eval.main(["--sources", str(sources), "--out", str(out)]) == 0
    census = json.loads((out / "census.json").read_text())
    assert set(census) == {"ovos-skill-demo", "ovos-skill-nogold"}
    assert census["ovos-skill-nogold"] == {"-": {"in": 0, "out": 0, "no_gold_file": 1}}
    assert "no_gold_file" not in json.dumps(census["ovos-skill-demo"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["test_rows"] == 2


def test_rows_land_per_locale_with_a_census(tmp_path):
    sources = _workspace(tmp_path)
    out = tmp_path / "eval"
    rc = build_eval.main(["--sources", str(sources), "--out", str(out)])
    assert rc == 0
    en = [json.loads(l) for l in (out / "en-US" / "test.jsonl").read_text().splitlines()]
    pt = [json.loads(l) for l in (out / "pt-PT" / "test.jsonl").read_text().splitlines()]
    assert [r["label"] for r in en] == ["ovos-skill-demo.openvoiceos:hello"]
    assert pt[0]["utterance"] == "olá" and pt[0]["lang"] == "pt-PT"
    census = json.loads((out / "census.json").read_text())
    cell = census["ovos-skill-demo"]["en-US"]
    assert cell["in"] == 4 and cell["out"] == 1
    assert cell["duplicate"] == 1 and cell["needs_manual"] == 1 and cell["no_intent"] == 1
    assert census["ovos-skill-demo"]["?"]["unparsable"] == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["test_rows"] == 2 and manifest["languages_test"] == 2
    assert manifest["label_function"] == "train/skill_labels.py"


def test_the_gate_refuses_a_gold_label_the_train_side_lacks(tmp_path):
    sources = _workspace(tmp_path)
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"label": "ovos-skill-demo.openvoiceos:goodbye",
                                 "lang": "en-US", "utterance": "bye"}) + "\n")
    rc = build_eval.main(["--sources", str(sources), "--out", str(tmp_path / "eval"),
                          "--train", str(train)])
    assert rc == 1


def test_the_gate_passes_when_every_gold_label_is_trained(tmp_path):
    sources = _workspace(tmp_path)
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"label": "ovos-skill-demo.openvoiceos:hello",
                                 "lang": "en-US", "utterance": "hi"}) + "\n")
    rc = build_eval.main(["--sources", str(sources), "--out", str(tmp_path / "eval"),
                          "--train", str(train)])
    assert rc == 0


def test_the_gate_is_the_census_module(monkeypatch, tmp_path):
    """build_eval calls census_gold_labels, it does not carry its own copy."""
    import census_gold_labels
    called = {}

    def fake(train_path, test_path):
        called["args"] = (train_path, test_path)
        return collections.Counter(), collections.Counter(), {}

    monkeypatch.setattr(build_eval, "census_paths", fake)
    sources = _workspace(tmp_path)
    train = tmp_path / "train.jsonl"
    train.write_text("")
    assert build_eval.census_paths is not census_gold_labels.census_paths
    build_eval.main(["--sources", str(sources), "--out", str(tmp_path / "eval"),
                     "--train", str(train)])
    assert called["args"][0] == train
