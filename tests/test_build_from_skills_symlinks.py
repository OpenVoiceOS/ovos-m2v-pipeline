"""A symbolically linked resource is read as the file it points at.

`git show` of a link returns the link's target, and the builder read that
as the file's content: ovos-skill-pokepedia links `pokemon_a.entity` to
`../vocab/pokemon.voc`, and the corpus carried 220 rows reading "battle
between ../vocab/pokemon.voc and ../vocab/pokemon.voc".
"""
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ovos_spec_tools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_from_skills as bfs  # noqa: E402

SETUP = '''
SKILL_NAME = "ovos-skill-demo"
PLUGIN_ENTRY_POINT = f"{SKILL_NAME}.openvoiceos=ovos_skill_demo:DemoSkill"
'''


def _skill(tmp_path, link_target):
    """A skill whose pokemon_a.entity is a link to *link_target*."""
    root = tmp_path / "ws" / "ovos" / "ovos-skill-demo"
    intents = root / "locale" / "en-US" / "intents"
    vocab = root / "locale" / "en-US" / "vocab"
    intents.mkdir(parents=True)
    vocab.mkdir(parents=True)
    (root / "setup.py").write_text(SETUP)
    (intents / "battle.intent").write_text("who wins between {pokemon_a}\n")
    (vocab / "pokemon.voc").write_text("pikachu\nbulbasaur\n")
    os.symlink(link_target, intents / "pokemon_a.entity")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t.invalid", "-c", "user.name=t",
                    "commit", "-q", "-m", "skill"], cwd=root, check=True)
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         check=True).stdout.strip()
    return root, sha


def _read(root, sha):
    stats = collections.Counter()
    templates, hints, _ = bfs.read_skill(root, sha, "ovos-skill-demo", stats,
                                         set(), [])
    return templates, hints, stats


def test_a_linked_entity_is_read_as_its_target(tmp_path):
    root, sha = _skill(tmp_path, "../vocab/pokemon.voc")
    _, hints, stats = _read(root, sha)
    assert hints["en-US"]["pokemon_a"] == ["pikachu", "bulbasaur"]
    assert stats["resource_read_through_a_symlink"] == 1


def test_a_link_chain_is_followed_to_the_file(tmp_path):
    root, sha = _skill(tmp_path, "pokemon_b.entity")
    intents = root / "locale" / "en-US" / "intents"
    os.symlink("../vocab/pokemon.voc", intents / "pokemon_b.entity")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t.invalid", "-c", "user.name=t",
                    "commit", "-q", "-m", "chain"], cwd=root, check=True)
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         check=True).stdout.strip()
    _, hints, _ = _read(root, sha)
    assert hints["en-US"]["pokemon_a"] == ["pikachu", "bulbasaur"]


@pytest.mark.parametrize("target,why", [
    ("../../../../../outside.voc", "outside the repository"),
    ("/etc/hostname", "outside the repository"),
    ("../vocab/missing.voc", "to a path the revision does not carry"),
    ("pokemon_a.entity", "symlink loop"),
])
def test_a_link_the_revision_does_not_carry_is_refused(tmp_path, target, why):
    root, sha = _skill(tmp_path, target)
    with pytest.raises(SystemExit, match=f"pokemon_a.entity.*{why}"):
        _read(root, sha)


def test_the_link_path_never_reaches_a_row(tmp_path):
    root, sha = _skill(tmp_path, "../vocab/pokemon.voc")
    templates, hints, _ = _read(root, sha)
    rows = [s for lang, _, t in templates
            for s in bfs.fill(t, hints[lang], {}, collections.Counter(), lang)]
    assert sorted(rows) == ["who wins between bulbasaur",
                            "who wins between pikachu"]


def test_the_build_refuses_a_row_carrying_a_resource_path(
        tmp_path, monkeypatch, capsys):
    """The gate reads finished rows, so it holds if link handling regresses."""
    root, sha = _skill(tmp_path, "../vocab/pokemon.voc")
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump({
        "workspace": str(tmp_path / "ws"),
        "skill_refs": {"refs": {"ovos/ovos-skill-demo": sha}}}))
    labels = tmp_path / "labels.yaml"
    labels.write_text(yaml.safe_dump({
        "version": 1,
        "baseline": {"corpus": "fixture", "labels_trained": 0, "labels": []},
        "removed": [], "added": []}))
    argv = ["build_from_skills.py", "--sources", str(sources),
            "--labels", str(labels), "--out", str(tmp_path / "out"),
            "--min-languages", "0", "--min-test-rows", "0",
            "--min-labels-scored", "0"]
    monkeypatch.setattr(sys, "argv", list(argv))
    assert bfs.main() == 0, capsys.readouterr().err
    rows = [json.loads(line)["utterance"] for line in
            (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert sorted(rows) == ["who wins between bulbasaur",
                            "who wins between pikachu"]

    # The defect re-planted: the link is read as a file again.
    monkeypatch.setattr(bfs, "read_resource",
                        lambda repo, rev, path, modes: bfs.show(repo, rev, path))
    monkeypatch.setattr(sys, "argv", list(argv))
    assert bfs.main() == 1
    assert "carry a resource path" in capsys.readouterr().err


@pytest.mark.parametrize("utterance,hit", [
    ("battle between ../vocab/pokemon.voc and pikachu", True),
    ("read pokemon_a.entity", True),
    ("open ../vocab/pokemon", True),
    ("say pokemon.voc twice", True),
    ("open the weather.intent file", True),
    ("play the dialogue from act two", False),
    ("what is 3.5 plus 2", False),
    ("go to example.com", False),
])
def test_the_path_detector(utterance, hit):
    row = {"lang": "en-US", "label": "x:y", "utterance": utterance}
    assert bool(bfs.rows_with_a_resource_path([row])) is hit
