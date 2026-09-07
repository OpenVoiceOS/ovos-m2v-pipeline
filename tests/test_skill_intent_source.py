"""A pinned skill's own `locale/<lang>/**/*.intent` files are a corpus source.

The external exports only carry the locales a translation round happened to
cover, so a skill that ships Kabyle or Swedish `.intent` files gets no rows
for them unless the skill tree itself is read.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_dataset  # noqa: E402


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


PYPROJECT = """
[project]
name = "ovos-skill-demo"
[project.entry-points."ovos.plugin.skill"]
"ovos-skill-demo.openvoiceos" = "x:y"
"""


@pytest.fixture
def demo(tmp_path):
    rev = git_repo(tmp_path / "skills" / "ovos-skill-demo", {
        "pyproject.toml": PYPROJECT,
        "locale/en-US/hello.intent": "# a comment line\nhello there\ngood (day|evening)\n",
        "locale/en-US/paint.intent": "paint it {colour}\n",
        "locale/en-US/colour.entity": "red\ngreen\n",
        "ovos_skill_demo/locale/kab-DZ/hello.intent": "azul\n",
        "locale/kab/hello.intent": "sbaḥ lxir\n",
        "test/end2end/fixture.intent": "not a registration\n",
    })
    return {"skill_refs": {"refs": {"skills/ovos-skill-demo": rev}}}


def rows_for(cfg, ws):
    rows, stats = [], __import__("collections").Counter()
    build_dataset.read_skill_intents(cfg, ws, rows, stats)
    return rows, stats


def test_every_locale_yields_labelled_rows(demo, tmp_path):
    rows, _ = rows_for(demo, tmp_path)
    by_lang = {}
    for lang, label, utt, source, _template in rows:
        by_lang.setdefault(lang, set()).add((label, utt))
        assert source == "skill-intents:ovos-skill-demo"

    assert by_lang["en-US"] == {
        ("ovos-skill-demo.openvoiceos:hello", "hello there"),
        ("ovos-skill-demo.openvoiceos:hello", "good day"),
        ("ovos-skill-demo.openvoiceos:hello", "good evening"),
        ("ovos-skill-demo.openvoiceos:paint", "paint it {colour}"),
    }
    # `locale/kab` and `<package>/locale/kab-DZ` are one language, and both
    # roots are read.
    assert set(by_lang) == {"en-US", "kab"}
    assert by_lang["kab"] == {
        ("ovos-skill-demo.openvoiceos:hello", "azul"),
        ("ovos-skill-demo.openvoiceos:hello", "sbaḥ lxir"),
    }


def test_comment_lines_and_test_trees_are_not_rows(demo, tmp_path):
    rows, _ = rows_for(demo, tmp_path)
    utterances = [r[2] for r in rows]
    assert not [u for u in utterances if u.startswith("#")]
    assert "not a registration" not in utterances


def test_slot_rows_fill_from_the_skills_own_entity_file(demo, tmp_path):
    from ovos_m2v_pipeline.slots import expand_entities
    cfg = {**demo, "git_sources": []}
    entities = build_dataset.collect_entities(cfg, tmp_path)
    assert entities["colour"] == ["red", "green"]
    assert set(expand_entities(["paint it {colour}"], entities)) == {
        "paint it red", "paint it green"}


def test_expansion_is_capped_per_line(tmp_path):
    line = " ".join(f"(a{i}|b{i})" for i in range(8))  # 256 combinations
    rev = git_repo(tmp_path / "skills" / "ovos-skill-demo", {
        "pyproject.toml": PYPROJECT,
        "locale/en-US/wide.intent": line + "\n",
    })
    rows, stats = rows_for({"skill_refs": {"refs": {"skills/ovos-skill-demo": rev}}},
                           tmp_path)
    assert len(rows) == build_dataset.TEMPLATE_EXPANSION_CAP
    assert stats["skill_intents_capped_templates"] == 1
