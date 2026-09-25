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
import json
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


# ------------------------------------------------------------ in flight ----
#
# Four parity tasks in a row (T-4107, T-4109, T-4111, T-4113) were filed for
# pairs an open pull request already carried. The census reads `origin/dev`,
# where a branch under review does not appear.


class _Ran:
    """A fake `gh` run, so no test reaches the network."""

    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def test_a_gap_an_open_pr_fills_is_in_flight_not_missing(tmp_path):
    repo = _skill(tmp_path, {
        "locale/en-US/a.intent": "one\n",
        "locale/en-US/b.intent": "two\n",
        "locale/pt-PT/a.intent": "um\n",
    })
    index = census.in_flight_index(
        [{"number": 31, "files": ["locale/pt-PT/b.intent"]}])
    row = census.layer_b(census.locale_map(census.files_at(repo, "origin/dev")),
                         "en-US", index)[0]
    assert row[".intent"]["missing"] == []
    assert row[".intent"]["in_flight"] == [{"name": "b", "pr": 31}]


def test_without_the_index_the_same_gap_is_missing(tmp_path):
    """The control. The pair is a gap; only an open pull request moves it."""
    repo = _skill(tmp_path, {
        "locale/en-US/a.intent": "one\n",
        "locale/en-US/b.intent": "two\n",
        "locale/pt-PT/a.intent": "um\n",
    })
    row = census.layer_b(census.locale_map(census.files_at(repo, "origin/dev")),
                         "en-US")[0]
    assert row[".intent"]["missing"] == ["b"]
    assert row[".intent"]["in_flight"] == []


def test_a_pr_touching_another_locale_does_not_cover_this_one(tmp_path):
    repo = _skill(tmp_path, {
        "locale/en-US/b.intent": "two\n",
        "locale/pt-PT/a.intent": "um\n",
    })
    index = census.in_flight_index(
        [{"number": 9, "files": ["locale/de-DE/b.intent"]}])
    row = [r for r in census.layer_b(
        census.locale_map(census.files_at(repo, "origin/dev")), "en-US", index)
        if r["lang"] == "pt-PT"][0]
    assert row[".intent"]["missing"] == ["b"]


def test_the_index_reads_any_locale_path_shape():
    """`locale/<lang>/intents/x.intent` counts the same as a flat path."""
    index = census.in_flight_index(
        [{"number": 4, "files": ["locale/eu-ES/intents/x.intent",
                                 "README.md",
                                 "test/test_x.py"]}])
    assert index == {("eu-ES", "x.intent"): 4}


def test_the_lowest_pull_request_number_wins():
    index = census.in_flight_index([
        {"number": 70, "files": ["locale/kab/x.dialog"]},
        {"number": 12, "files": ["locale/kab/x.dialog"]},
    ])
    assert index[("kab", "x.dialog")] == 12


def test_open_pr_files_reads_the_paths():
    calls = []

    def runner(cmd):
        calls.append(cmd)
        return _Ran('[{"number": 5, "headRefOid": "abc", '
                    '"files": [{"path": "locale/it-IT/x.voc"}]}]')

    assert census.open_pr_files("Org/skill", runner) == [
        {"number": 5, "head": "abc", "files": ["locale/it-IT/x.voc"],
         "refetched": False}]
    assert "--state" in calls[0] and "open" in calls[0]


def test_a_full_file_list_is_read_again_from_the_paging_api():
    """`gh pr list` caps at 100 files and says nothing about it.

    moviemaster#105 carries 85 locale files inside its first 100 and more
    after them, so a census that trusted the cap would call those pairs
    missing and file a task for work already written.
    """
    big = [{"path": f"locale/ca-ES/f{i}.dialog"} for i in range(100)]
    seen = []

    def runner(cmd):
        seen.append(cmd)
        if cmd[1] == "pr":
            return _Ran(json.dumps([{"number": 105, "headRefOid": "a",
                                     "files": big}]))
        return _Ran("locale/ca-ES/f0.dialog\nlocale/ca-ES/late.dialog\n")

    prs = census.open_pr_files("Org/skill", runner)
    assert prs[0]["refetched"] is True
    assert "locale/ca-ES/late.dialog" in prs[0]["files"]
    assert seen[1][:2] == ["gh", "api"] and "--paginate" in seen[1]


def test_a_short_file_list_is_not_read_again():
    """The control: no second call when nothing was cut."""
    seen = []

    def runner(cmd):
        seen.append(cmd)
        return _Ran(json.dumps([{"number": 7, "headRefOid": "a",
                                 "files": [{"path": "locale/it-IT/x.voc"}]}]))

    prs = census.open_pr_files("Org/skill", runner)
    assert prs[0]["refetched"] is False
    assert len(seen) == 1


def test_a_gh_failure_is_raised_and_not_read_as_no_open_prs():
    """Silence here would report every gap as a gap with no warning."""
    with pytest.raises(RuntimeError):
        census.open_pr_files("Org/skill",
                             lambda cmd: _Ran("", 1, "could not resolve host"))


def test_the_slug_comes_from_the_origin_remote(tmp_path):
    repo = _skill(tmp_path, {"locale/en-US/a.intent": "one\n"})
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin",
                    "https://github.com/OpenVoiceOS/ovos-skill-fake.git"],
                   check=True, capture_output=True)
    assert census.repo_slug(repo) == "OpenVoiceOS/ovos-skill-fake"


# ------------------------------------------------------- branch signal ----

def _push_branch(repo: Path, name: str, files: dict) -> None:
    """Commit `files` on `name` and leave it only as `origin/<name>`.

    The work must not be on `origin/dev`, which is what the census reads.
    """
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,
                                    capture_output=True)
    run("checkout", "-q", "-b", name, "origin/dev")
    for path, text in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", name)
    run("update-ref", f"refs/remotes/origin/{name}", "HEAD")
    run("checkout", "-q", "dev")
    run("reset", "-q", "--hard", "origin/dev")


def test_a_pair_on_a_pushed_branch_with_no_pr_is_in_flight(tmp_path):
    repo = _skill(tmp_path, {
        "locale/en-US/a.intent": "one\n",
        "locale/en-US/b.intent": "two\n",
        "locale/pt-PT/a.intent": "um\n",
    })
    _push_branch(repo, "t4108-parity", {"locale/pt-PT/b.intent": "dois\n"})
    index = census.branch_index(repo, "origin/dev")
    assert index == {("pt-PT", "b.intent"): "branch:t4108-parity"}
    row = census.layer_b(census.locale_map(census.files_at(repo, "origin/dev")),
                         "en-US", index)[0]
    assert row[".intent"]["missing"] == []
    assert row[".intent"]["in_flight"] == [{"name": "b",
                                            "branch": "t4108-parity"}]


def test_without_the_branch_signal_the_same_pair_is_missing(tmp_path):
    """The control. The branch exists; only reading it moves the pair."""
    repo = _skill(tmp_path, {
        "locale/en-US/a.intent": "one\n",
        "locale/en-US/b.intent": "two\n",
        "locale/pt-PT/a.intent": "um\n",
    })
    _push_branch(repo, "t4108-parity", {"locale/pt-PT/b.intent": "dois\n"})
    row = census.layer_b(census.locale_map(census.files_at(repo, "origin/dev")),
                         "en-US")[0]
    assert row[".intent"]["missing"] == ["b"]
    assert row[".intent"]["in_flight"] == []


def test_a_branch_that_is_not_a_task_branch_is_not_read(tmp_path):
    """`t<digits>-` is the rule. A release or a feature branch is not work
    of this lane and its files must not silence a gap."""
    repo = _skill(tmp_path, {
        "locale/en-US/b.intent": "two\n",
        "locale/pt-PT/a.intent": "um\n",
    })
    _push_branch(repo, "feat/whatever", {"locale/pt-PT/b.intent": "dois\n"})
    assert census.task_branches(repo) == []
    assert census.branch_index(repo, "origin/dev") == {}


def test_a_branch_touching_another_locale_does_not_cover_this_one(tmp_path):
    repo = _skill(tmp_path, {
        "locale/en-US/b.intent": "two\n",
        "locale/pt-PT/a.intent": "um\n",
    })
    _push_branch(repo, "t1-parity", {"locale/de-DE/b.intent": "zwei\n"})
    index = census.branch_index(repo, "origin/dev")
    row = [r for r in census.layer_b(
        census.locale_map(census.files_at(repo, "origin/dev")), "en-US", index)
        if r["lang"] == "pt-PT"][0]
    assert row[".intent"]["missing"] == ["b"]


def test_a_branch_merged_into_dev_adds_nothing(tmp_path):
    """A branch whose work is already on dev has no pair to report, so the
    signal cannot invent one."""
    repo = _skill(tmp_path, {
        "locale/en-US/b.intent": "two\n",
        "locale/pt-PT/b.intent": "dois\n",
    })
    _push_branch(repo, "t2-parity", {"README.md": "x\n"})
    assert census.branch_index(repo, "origin/dev") == {}


def test_the_first_branch_in_name_order_wins(tmp_path):
    repo = _skill(tmp_path, {"locale/en-US/b.intent": "two\n"})
    _push_branch(repo, "t9-late", {"locale/kab/b.intent": "a\n"})
    _push_branch(repo, "t1-early", {"locale/kab/b.intent": "b\n"})
    index = census.branch_index(repo, "origin/dev")
    assert index[("kab", "b.intent")] == "branch:t1-early"


def test_a_pull_request_number_wins_over_a_branch():
    merged = census.merge_signals({("kab", "x.voc"): 12},
                                  {("kab", "x.voc"): "branch:t1-x",
                                   ("kab", "y.voc"): "branch:t1-x"})
    assert merged[("kab", "x.voc")] == 12
    assert merged[("kab", "y.voc")] == "branch:t1-x"


def _skill_with_remote(tmp_path: Path, files: dict, branch_files: dict,
                       branch: str = "t4108-kab-parity") -> Path:
    """A clone whose remote holds a task branch the clone does NOT hold.

    This is the real shape: on 2026-09-25 every task branch this lane held
    was on the remote and in no clone, so a reader of local refs saw none.
    """
    run = lambda where, *a: subprocess.run(["git", "-C", str(where), *a],
                                           check=True, capture_output=True)
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q", "--bare", "-b", "dev", str(origin)],
                   check=True, capture_output=True)
    run(work, "init", "-q", "-b", "dev")
    run(work, "config", "user.email", "t@t")
    run(work, "config", "user.name", "t")
    for name, text in files.items():
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    run(work, "add", "-A")
    run(work, "commit", "-q", "-m", "dev")
    run(work, "remote", "add", "origin", str(origin))
    run(work, "push", "-q", "origin", "dev")
    run(work, "checkout", "-q", "-b", branch)
    for name, text in branch_files.items():
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    run(work, "add", "-A")
    run(work, "commit", "-q", "-m", branch)
    run(work, "push", "-q", "origin", branch)

    clone = tmp_path / "ovos-skill-fake"
    subprocess.run(["git", "clone", "-q", "--single-branch", "--branch", "dev",
                    str(origin), str(clone)], check=True, capture_output=True)
    return clone


def test_a_branch_only_on_the_remote_is_read_when_fetched(tmp_path):
    clone = _skill_with_remote(
        tmp_path,
        {"locale/en-US/a.intent": "one\n", "locale/en-US/b.intent": "two\n",
         "locale/pt-PT/a.intent": "um\n"},
        {"locale/pt-PT/b.intent": "dois\n"})
    assert census.task_branches(clone) == []            # not in the clone
    assert census.remote_task_branches(clone) == ["t4108-kab-parity"]
    index = census.branch_index(clone, "origin/dev", fetch=True)
    assert index == {("pt-PT", "b.intent"): "branch:t4108-kab-parity"}


def test_the_same_branch_is_invisible_without_the_fetch(tmp_path):
    """The control, and the reason the fetch is the default: reading local
    refs alone answers empty for a branch that exists."""
    clone = _skill_with_remote(
        tmp_path,
        {"locale/en-US/a.intent": "one\n", "locale/en-US/b.intent": "two\n",
         "locale/pt-PT/a.intent": "um\n"},
        {"locale/pt-PT/b.intent": "dois\n"})
    assert census.branch_index(clone, "origin/dev", fetch=False) == {}


def test_a_branch_that_cannot_be_fetched_is_named_not_skipped(tmp_path):
    repo = _skill(tmp_path, {"locale/en-US/b.intent": "two\n"})
    notes: list = []
    index = census.branch_index(repo, "origin/dev", branches=["t9-gone"],
                                fetch=True, notes=notes)
    assert index == {}
    assert notes and notes[0]["branch"] == "t9-gone"


def test_a_force_pushed_branch_is_still_read(tmp_path):
    """The defect the alerts fleet run found.

    `ovos-skill-alerts` held a stale `origin/t4213-slot-parity`, because the
    branch was force-pushed after the clone last fetched it. The plain
    refspec fails as a non-fast-forward, so the branch dropped out of the
    signal and its pairs read as gaps. The forced refspec follows the
    remote, which is what a reader sees.
    """
    clone = _skill_with_remote(
        tmp_path,
        {"locale/en-US/a.intent": "one\n", "locale/en-US/b.intent": "two\n",
         "locale/pt-PT/a.intent": "um\n"},
        {"locale/pt-PT/b.intent": "dois\n"})
    run = lambda where, *a: subprocess.run(["git", "-C", str(where), *a],
                                           check=True, capture_output=True)
    # The clone holds the first push of the branch.
    census.branch_index(clone, "origin/dev", fetch=True)
    first = subprocess.run(
        ["git", "-C", str(clone), "rev-parse",
         "refs/remotes/origin/t4108-kab-parity"],
        check=True, capture_output=True, text=True).stdout.strip()

    # The branch is rewritten and force-pushed, the shape a squash makes.
    work = tmp_path / "work"
    run(work, "checkout", "-q", "t4108-kab-parity")
    run(work, "reset", "-q", "--hard", "dev")
    (work / "locale/pt-PT/b.intent").write_text("dois\n", encoding="utf-8")
    (work / "locale/pt-PT/c.intent").write_text("tres\n", encoding="utf-8")
    run(work, "add", "-A")
    run(work, "commit", "-q", "-m", "squashed")
    run(work, "push", "-q", "--force", "origin", "t4108-kab-parity")

    notes: list = []
    index = census.branch_index(clone, "origin/dev", fetch=True, notes=notes)
    assert notes == []
    assert index == {("pt-PT", "b.intent"): "branch:t4108-kab-parity",
                     ("pt-PT", "c.intent"): "branch:t4108-kab-parity"}
    after = subprocess.run(
        ["git", "-C", str(clone), "rev-parse",
         "refs/remotes/origin/t4108-kab-parity"],
        check=True, capture_output=True, text=True).stdout.strip()
    assert after != first
