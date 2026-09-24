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


def test_the_report_names_the_listing_it_counted(tmp_path, capsys):
    """51 in, 51 out is a census of the workspace; the report must say which."""
    root = tmp_path / "ws" / "ovos" / "skills"
    root.mkdir(parents=True)
    _skill(root, {"locale/en-US/one.dialog": "hello\n"})
    census.main(["--workspace", str(tmp_path / "ws")])
    out = capsys.readouterr().out
    assert f"listing: {tmp_path / 'ws' / 'ovos/skills/ovos-skill-*'} (1 clones)" in out
    assert "fleet: not checked" in out


def test_a_fleet_name_with_no_clone_is_named_not_cloned(tmp_path, capsys):
    """A skill the workspace does not hold produces no row; the report says so."""
    root = tmp_path / "ws" / "ovos" / "skills"
    root.mkdir(parents=True)
    _skill(root, {"locale/en-US/one.dialog": "hello\n"})
    listing = tmp_path / "fleet.txt"
    listing.write_text("# the org\novos-skill-fake\novos-skill-news\n"
                       "OpenVoiceOS/ovos-skill-spotify\n", encoding="utf-8")
    code = census.main(["--workspace", str(tmp_path / "ws"),
                        "--fleet", str(listing)])
    out = capsys.readouterr().out
    assert "1 skills in, 1 rows out" in out
    assert "names 3, 2 not cloned, 0 clones not in the listing" in out
    assert "  NOT CLONED ovos-skill-news\n" in out
    assert "  NOT CLONED ovos-skill-spotify\n" in out
    assert "NOT IN FLEET" not in out
    assert code == 0, "a name the census did not measure is not a Layer A error"


def test_a_clone_the_listing_lacks_is_named_not_in_fleet(tmp_path, capsys):
    root = tmp_path / "ws" / "ovos" / "skills"
    root.mkdir(parents=True)
    _skill(root, {"locale/en-US/one.dialog": "hello\n"})
    listing = tmp_path / "fleet.txt"
    listing.write_text("ovos-skill-news\n", encoding="utf-8")
    census.main(["--workspace", str(tmp_path / "ws"), "--fleet", str(listing)])
    out = capsys.readouterr().out
    assert "names 1, 1 not cloned, 1 clones not in the listing" in out
    assert "  NOT IN FLEET ovos-skill-fake\n" in out


def test_the_json_report_carries_the_listing_and_the_fleet_diff(tmp_path, capsys):
    import json
    root = tmp_path / "ws" / "ovos" / "skills"
    root.mkdir(parents=True)
    _skill(root, {"locale/en-US/one.dialog": "hello\n"})
    listing = tmp_path / "fleet.txt"
    listing.write_text("ovos-skill-news\novos-skill-fake\n", encoding="utf-8")
    census.main(["--workspace", str(tmp_path / "ws"), "--fleet", str(listing),
                 "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["listing"].endswith("ovos/skills/ovos-skill-*")
    assert report["fleet"]["not_cloned"] == ["ovos-skill-news"]
    assert report["fleet"]["not_in_fleet"] == []
    assert report["fleet"]["named"] == 2


def test_an_archived_org_repository_is_excluded_from_the_gaps(tmp_path, capsys,
                                                               monkeypatch):
    """An archived repository is not a gap the workspace can close by cloning
    it, so ``--org`` must drop it before the comparison, and the report must
    say how many it dropped.
    """
    root = tmp_path / "ws" / "ovos" / "skills"
    root.mkdir(parents=True)
    _skill(root, {"locale/en-US/one.dialog": "hello\n"})

    class FakeGh:
        returncode = 0
        stdout = ("ovos-skill-fake\tfalse\n"
                  "ovos-skill-retired\ttrue\n")
        stderr = ""

    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["gh", "api"]:
            return FakeGh()
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(census.subprocess, "run", fake_run)
    census.main(["--workspace", str(tmp_path / "ws"), "--org", "OpenVoiceOS"])
    out = capsys.readouterr().out
    assert "NOT CLONED ovos-skill-retired" not in out
    assert "1 archived repositories excluded from the listing" in out


def test_org_and_fleet_together_are_refused(tmp_path):
    root = tmp_path / "ws" / "ovos" / "skills"
    root.mkdir(parents=True)
    _skill(root, {"locale/en-US/one.dialog": "hello\n"})
    with pytest.raises(SystemExit) as raised:
        census.main(["--workspace", str(tmp_path / "ws"), "--org", "x",
                     "--fleet", str(tmp_path / "none.txt")])
    assert "not both" in str(raised.value)
