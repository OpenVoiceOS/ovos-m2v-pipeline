"""The census must measure what the specifications say, and nothing more.

Three properties matter more than the numbers it prints:

* Layer A comes from the linter. If this module re-implemented a rule, the
  census could pass while the linter fails, and a skill would ship a defect
  the gate says is not there.
* A `{slot}` with no `<slot>.entity` is inventory, never a violation
  (OVOS-INTENT-1 §5.4).
* Layer B is policy. No clause requires a locale to mirror `en-US`
  (locale-parity.md §1), so its rows must never raise the exit status.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import parity_census as census  # noqa: E402


def _skill(tmp_path: Path, files: dict) -> Path:
    """A git repository with a locale tree, committed on a dev branch."""
    repo = tmp_path / "ovos-skill-fake"
    repo.mkdir()
    for name, text in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,
                                    capture_output=True)
    run("init", "-q", "-b", "dev")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    run("add", "-A")
    run("commit", "-q", "-m", "x")
    run("update-ref", "refs/remotes/origin/dev", "HEAD")
    return repo


def test_a_slot_without_an_entity_is_inventory_not_a_finding(tmp_path):
    repo = _skill(tmp_path, {
        "locale/en-US/play.intent": "play {query}\n",
    })
    out = census.census_one(repo, "origin/dev")
    rows = [r for r in out["slots"] if r.get("slot") == "query"]
    assert rows and rows[0]["entity"] is False
    assert not [f for f in out["layer_a"]
                if "entity" in f["message"] and f["severity"] == "error"]


def test_a_typed_slot_is_reported_under_its_name(tmp_path):
    """OVOS-INTENT-1 §5.6; declared_slots does this, not a regex here."""
    repo = _skill(tmp_path, {
        "locale/en-US/count.intent": "count to {number:target}\n",
    })
    out = census.census_one(repo, "origin/dev")
    assert {r["slot"] for r in out["slots"] if r.get("slot")} == {"target"}


def test_layer_b_reports_a_missing_file_against_the_reference(tmp_path):
    repo = _skill(tmp_path, {
        "locale/en-US/one.dialog": "hello\n",
        "locale/en-US/two.dialog": "goodbye\n",
        "locale/de-DE/one.dialog": "hallo\n",
    })
    out = census.census_one(repo, "origin/dev")
    de = [row for row in out["layer_b"] if row["lang"] == "de-DE"][0]
    assert de[".dialog"]["missing"] == ["two"]
    assert de[".dialog"]["extra"] == []


def test_layer_b_reports_an_extra_file_without_calling_it_missing(tmp_path):
    """A locale-only file is allowed by silence (locale-parity.md §5 Q7)."""
    repo = _skill(tmp_path, {
        "locale/en-US/one.dialog": "hello\n",
        "locale/de-DE/one.dialog": "hallo\n",
        "locale/de-DE/extra.dialog": "extra\n",
    })
    out = census.census_one(repo, "origin/dev")
    de = [row for row in out["layer_b"] if row["lang"] == "de-DE"][0]
    assert de[".dialog"]["missing"] == []
    assert de[".dialog"]["extra"] == ["extra"]


def test_a_skill_with_no_reference_locale_gets_no_layer_b(tmp_path):
    """The control: mirroring against a locale that is not there is nothing."""
    repo = _skill(tmp_path, {"locale/de-DE/one.dialog": "hallo\n"})
    out = census.census_one(repo, "origin/dev")
    assert out["reference_present"] is False
    assert out["layer_b"] == []


def test_layer_a_findings_come_from_the_linter(tmp_path):
    """A name the linter refuses must appear, with the linter's own words."""
    repo = _skill(tmp_path, {"locale/en-US/NotLowercase.dialog": "hello\n"})
    out = census.census_one(repo, "origin/dev")
    messages = [f["message"] for f in out["layer_a"]]
    assert any("base name" in m and "NotLowercase" in m for m in messages), messages


def test_a_clean_skill_has_no_layer_a_error(tmp_path):
    """The control in the other direction: the check is not always positive."""
    repo = _skill(tmp_path, {
        "locale/en-US/greet.dialog": "hello there\n",
        "locale/en-US/greet.intent": "say hello to {name}\n",
    })
    out = census.census_one(repo, "origin/dev")
    assert [f for f in out["layer_a"] if f["severity"] == "error"] == []


def test_the_blacklist_column_pairs_by_base_name(tmp_path):
    """OVOS-INTENT-2 §4.3, not the linter's blacklist differential."""
    repo = _skill(tmp_path, {
        "locale/en-US/play.intent": "play {query}\n",
        "locale/en-US/query.entity": "jazz\n",
        "locale/en-US/query.blacklist": "it\n",
    })
    out = census.census_one(repo, "origin/dev")
    row = [r for r in out["slots"] if r.get("slot") == "query"][0]
    assert row["entity"] is True
    assert row["blacklist"] is True


def test_the_sha_is_recorded(tmp_path):
    repo = _skill(tmp_path, {"locale/en-US/one.dialog": "hello\n"})
    out = census.census_one(repo, "origin/dev")
    assert len(out["sha"]) == 40


def test_the_exit_status_is_one_on_a_layer_a_error(tmp_path, capsys):
    root = tmp_path / "ws" / "ovos" / "skills"
    root.mkdir(parents=True)
    _skill(root, {"locale/en-US/NotLowercase.dialog": "hello\n"})
    code = census.main(["--workspace", str(tmp_path / "ws")])
    assert code == 1


def test_the_exit_status_is_zero_when_only_layer_b_differs(tmp_path):
    """Policy rows must never fail a build; no clause requires mirroring."""
    root = tmp_path / "ws" / "ovos" / "skills"
    root.mkdir(parents=True)
    _skill(root, {
        "locale/en-US/one.dialog": "hello\n",
        "locale/en-US/two.dialog": "goodbye\n",
        "locale/de-DE/one.dialog": "hallo\n",
    })
    assert census.main(["--workspace", str(tmp_path / "ws")]) == 0


def test_every_skill_in_is_a_row_out(tmp_path, capsys):
    root = tmp_path / "ws" / "ovos" / "skills"
    root.mkdir(parents=True)
    for name in ("a", "b", "c"):
        repo = _skill(root, {"locale/en-US/one.dialog": "hello\n"})
        repo.rename(repo.parent / f"ovos-skill-{name}")
    census.main(["--workspace", str(tmp_path / "ws")])
    out = capsys.readouterr().out
    assert "3 skills in, 3 rows out" in out


def test_a_repository_without_dev_is_read_at_master(tmp_path, capsys):
    """The census must not drop a repository over the name of its branch."""
    root = tmp_path / "ws" / "ovos" / "skills"
    root.mkdir(parents=True)
    repo = _skill(root, {"locale/en-US/one.dialog": "hello\n"})
    subprocess.run(["git", "-C", str(repo), "update-ref", "-d",
                    "refs/remotes/origin/dev"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "update-ref",
                    "refs/remotes/origin/master", "HEAD"], check=True,
                   capture_output=True)
    census.main(["--workspace", str(tmp_path / "ws")])
    assert "1 skills in, 1 rows out" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# OVOS-INTENT-2 §2.3 (architecture#272, NOT merged: read at head 32f0973).
# Two defect classes Layer B does not report, and the two exceptions the
# clause carries. These tests are what says the clause was read, not guessed.
# ---------------------------------------------------------------------------


def test_2_3_reports_an_extra_file_against_the_locale_that_lacks_it(tmp_path):
    """Layer B calls this a locale-specific addition. §2.3 calls it a defect.

    The direction is what changed: a pair in ANY locale must be in EVERY
    other, so the file de-DE has and en-US lacks is reported against en-US.
    """
    repo = _skill(tmp_path, {
        "locale/en-US/play.intent": "play {query}\n",
        "locale/de-DE/play.intent": "spiele {query}\n",
        "locale/de-DE/extra.dialog": "extra\n",
    })
    out = census.census_one(repo, "origin/dev")
    rows = {r["lang"]: r["missing"] for r in out["parity_defects_2_3"]}
    assert "en-US" in rows, out["parity_defects_2_3"]
    assert {"role": ".dialog", "name": "extra"} in rows["en-US"]
    assert "de-DE" not in rows

    # and Layer B still calls it an extra, so the two reports stay apart
    layer_b = {r["lang"]: r for r in out["layer_b_vs_en_us"]}
    assert layer_b["de-DE"][".dialog"]["extra"] == ["extra"]
    assert layer_b["de-DE"][".dialog"]["missing"] == []


def test_2_3_excepts_blacklist_and_prompt(tmp_path):
    """A blacklist is a property of one language; a prompt is model input."""
    repo = _skill(tmp_path, {
        "locale/en-US/play.intent": "play {query}\n",
        "locale/de-DE/play.intent": "spiele {query}\n",
        "locale/de-DE/pronoun.blacklist": "er\n",
    })
    out = census.census_one(repo, "origin/dev")
    for row in out["parity_defects_2_3"]:
        assert all(m["role"] != ".blacklist" for m in row["missing"]), row


def test_2_3_a_voc_an_intent_references_inline_is_not_a_parity_gap(tmp_path):
    """§2.3: one file is never both a parity gap and a malformed-file error.

    en-US ships `colour.voc` and de-DE does not, which is a parity gap —
    UNLESS a de-DE `.intent` carries `<colour>` inline. Then that `.intent`
    is malformed under OVOS-INTENT-1 §3.6 and the linter owns it.
    """
    repo = _skill(tmp_path, {
        "locale/en-US/play.intent": "play {query}\n",
        "locale/en-US/colour.voc": "red\n",
        "locale/de-DE/play.intent": "spiele <colour> {query}\n",
    })
    out = census.census_one(repo, "origin/dev")
    rows = {r["lang"]: r["missing"] for r in out["parity_defects_2_3"]}
    assert {"role": ".voc", "name": "colour"} not in rows.get("de-DE", [])

    # the control: with no inline reference, the same gap IS a parity defect
    (tmp_path / "b").mkdir()
    other = _skill(tmp_path / "b", {
        "locale/en-US/play.intent": "play {query}\n",
        "locale/en-US/colour.voc": "red\n",
        "locale/de-DE/play.intent": "spiele {query}\n",
    })
    out2 = census.census_one(other, "origin/dev")
    rows2 = {r["lang"]: r["missing"] for r in out2["parity_defects_2_3"]}
    assert {"role": ".voc", "name": "colour"} in rows2["de-DE"]


def test_2_3_reports_a_slot_set_difference_by_intent_name(tmp_path):
    """The union of the slot names one intent declares, per locale."""
    repo = _skill(tmp_path, {
        "locale/en-US/play.intent": "play {query} by {artist}\n",
        "locale/de-DE/play.intent": "spiele {query}\n",
    })
    out = census.census_one(repo, "origin/dev")
    rows = {r["intent"]: r for r in out["slot_set_defects_2_3"]}
    assert "play" in rows, out["slot_set_defects_2_3"]
    assert rows["play"]["union"] == ["artist", "query"]
    assert rows["play"]["missing_per_lang"] == {"de-DE": ["artist"]}


def test_2_3_an_identical_slot_set_is_not_reported(tmp_path):
    repo = _skill(tmp_path, {
        "locale/en-US/play.intent": "play {query}\n",
        "locale/de-DE/play.intent": "spiele {query}\n",
    })
    out = census.census_one(repo, "origin/dev")
    assert out["slot_set_defects_2_3"] == []
