"""A label leaves the corpus only when the ledger says why.

The floor this replaces was one number. A number cannot tell a label that
left on purpose from a label whose resource was deleted by mistake, and the
only repair it offers is to lower itself. Worse, it is a net: a label lost on
the same day another is gained moves the total by nothing, so the loss never
reaches the number at all. These tests hold the two properties that replace
it -- the floor is derived from the listed changes and never typed, and an
unlisted loss fails the build whatever the total says.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_from_skills as bfs  # noqa: E402

LEDGER = Path(__file__).resolve().parents[1] / "train" / "labels.yaml"


def write(tmp_path, doc):
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def small(**over):
    doc = {
        "version": 1,
        "baseline": {"corpus": "test", "labels_trained": 3,
                     "labels": ["s:a", "s:b", "s:c"]},
        "removed": [{"label": "s:b", "reason": "folded into s:a",
                     "merged_into": "s:a"}],
        "added": [{"label": "s:d", "reason": "a new intent"}],
    }
    doc.update(over)
    return doc


def uneven(**over):
    """A fixture whose derived floor cannot equal the bare baseline count.

    ``small()`` has one removal and one addition, so 3 - 1 + 1 is 3, which is
    also the baseline count: a test on that fixture passes whether the floor
    is derived or simply read off the baseline. Two removals and one addition
    give 2, which only the derivation produces.
    """
    doc = {
        "version": 1,
        "baseline": {"corpus": "test", "labels_trained": 3,
                     "labels": ["s:a", "s:b", "s:c"]},
        "removed": [{"label": "s:b", "reason": "folded into s:a",
                     "merged_into": "s:a"},
                    {"label": "s:c", "reason": "folded into s:a",
                     "merged_into": "s:a"}],
        "added": [{"label": "s:d", "reason": "a new intent"}],
    }
    doc.update(over)
    return doc


def test_the_floor_is_derived_from_the_listed_changes(tmp_path):
    ledger = bfs.load_label_ledger(write(tmp_path, uneven()))
    assert ledger["min_labels"] == 3 - 2 + 1
    assert ledger["min_labels"] != ledger["baseline_count"]


def test_a_pending_addition_does_not_raise_the_floor(tmp_path):
    """reviewer-skills' control on #281: the floor reads the pins, not the file.

    The ledger lists a removal that HAS taken effect and an addition that has
    NOT arrived. Deriving the floor from the whole ledger gives 3 - 2 + 1 = 2
    while the corpus holds 1 baseline label, so the build fails on a state the
    ledger documents as safe. The floor is the size of the expected set, and
    s:d is not expected until it exists.
    """
    ledger = bfs.load_label_ledger(write(tmp_path, uneven()))
    trained = {"s:a"}          # s:b and s:c removed as listed, s:d not here yet
    out = bfs.check_label_ledger(ledger, trained)
    assert out["listed_additions_not_yet_arrived"] == ["s:d"]
    assert out["labels_lost_and_unlisted"] == []
    assert out["min_labels_derived"] == 1
    assert len(trained) >= out["min_labels_derived"]
    # and the whole-ledger figure, which the manifest still reports, is higher
    assert out["min_labels_if_every_listed_change_lands"] == 2


def test_an_arrived_addition_does_raise_the_floor(tmp_path):
    """The other half: once the label is here, losing it is a shortfall."""
    ledger = bfs.load_label_ledger(write(tmp_path, uneven()))
    out = bfs.check_label_ledger(ledger, {"s:a", "s:d"})
    assert out["listed_additions_not_yet_arrived"] == []
    assert out["min_labels_derived"] == 2


def test_a_removal_not_yet_applied_holds_the_floor_up(tmp_path):
    """A listed removal moves the floor when it happens, not when it is written."""
    ledger = bfs.load_label_ledger(write(tmp_path, uneven()))
    out = bfs.check_label_ledger(ledger, {"s:a", "s:b", "s:c"})
    assert out["listed_removals_not_yet_applied"] == ["s:b", "s:c"]
    assert out["min_labels_derived"] == 3


def test_a_listed_loss_is_accepted(tmp_path):
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    out = bfs.check_label_ledger(ledger, {"s:a", "s:c", "s:d"})
    assert out["labels_lost_and_unlisted"] == []
    assert out["labels_lost_against_baseline"] == ["s:b"]


def test_an_unlisted_loss_is_reported(tmp_path):
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    out = bfs.check_label_ledger(ledger, {"s:a", "s:b", "s:d"})
    assert out["labels_lost_and_unlisted"] == ["s:c"]


def test_an_unlisted_loss_is_reported_even_when_the_total_holds(tmp_path):
    """The failure a net count cannot see: one lost, one gained, total flat."""
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    trained = {"s:a", "s:b", "s:d"}
    assert len(trained) == ledger["min_labels"]
    out = bfs.check_label_ledger(ledger, trained)
    assert out["labels_lost_and_unlisted"] == ["s:c"]


def test_a_removal_whose_survivor_is_missing_is_reported(tmp_path):
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    out = bfs.check_label_ledger(ledger, {"s:c", "s:d"})
    assert out["merge_targets_missing_from_the_corpus"] == ["s:a"]


def test_a_removal_not_yet_applied_is_reported_and_is_not_a_loss(tmp_path):
    """The ledger may lead the pins, so this is a note and not a failure."""
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    out = bfs.check_label_ledger(ledger, {"s:a", "s:b", "s:c", "s:d"})
    assert out["listed_removals_not_yet_applied"] == ["s:b"]
    assert out["labels_lost_and_unlisted"] == []


def test_a_removal_needs_a_reason(tmp_path):
    doc = small(removed=[{"label": "s:b", "reason": "  ", "merged_into": None}])
    with pytest.raises(SystemExit):
        bfs.load_label_ledger(write(tmp_path, doc))


def test_a_removal_needs_to_say_where_the_phrasings_went(tmp_path):
    doc = small(removed=[{"label": "s:b", "reason": "gone"}])
    with pytest.raises(SystemExit):
        bfs.load_label_ledger(write(tmp_path, doc))


def test_a_removal_of_a_label_the_baseline_never_had_is_refused(tmp_path):
    doc = small(removed=[{"label": "s:z", "reason": "gone",
                          "merged_into": None}])
    with pytest.raises(SystemExit):
        bfs.load_label_ledger(write(tmp_path, doc))


def test_a_baseline_whose_count_disagrees_with_its_list_is_refused(tmp_path):
    doc = small()
    doc["baseline"]["labels_trained"] = 4
    with pytest.raises(SystemExit):
        bfs.load_label_ledger(write(tmp_path, doc))


def test_the_shipped_ledger_loads_and_derives_its_floor():
    ledger = bfs.load_label_ledger(LEDGER)
    assert ledger["baseline_count"] == 232
    assert ledger["min_labels"] == (
        232 - len(ledger["removed"]) + len(ledger["added"]))


def test_every_shipped_removal_names_a_label_the_baseline_carried():
    ledger = bfs.load_label_ledger(LEDGER)
    assert set(ledger["removed"]) <= ledger["baseline_labels"]


# --------------------------------------------------------------------------
# The gate, run end to end.
#
# The checks above call the two functions directly. These drive the real
# builder over a fixture workspace and assert its exit code, because the
# defect this round repairs was not in either function on its own: it was in
# the floor main() carried from the ledger file to the gate. reviewer-skills
# measured it that way on #281 and the control is kept here so it stays
# measured.
# --------------------------------------------------------------------------
BUILDER = Path(__file__).resolve().parents[1] / "train" / "build_from_skills.py"


def _skill_repo(root, intents):
    """A throwaway skill repo: base name -> one template line."""
    for name, line in intents.items():
        path = root / "locale" / "en-US" / f"{name}.intent"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(line + "\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@test.invalid", "-c", "user.name=test",
         "commit", "-q", "-m", "skill"], cwd=root, check=True)
    return root


def _run_build(tmp_path, intents, ledger_doc, extra=()):
    """Build a one-skill corpus and return (rc, stderr)."""
    ws = tmp_path / "ws"
    repo = ws / "OpenVoiceOS" / "ovos-skill-control"
    repo.mkdir(parents=True)
    _skill_repo(repo, intents)
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump(
        {"workspace": str(ws),
         "skill_refs": {"refs": {"OpenVoiceOS/ovos-skill-control": "HEAD"}}},
        sort_keys=False), encoding="utf-8")
    labels = tmp_path / "labels.yaml"
    labels.write_text(yaml.safe_dump(ledger_doc, sort_keys=False),
                      encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(BUILDER),
         "--sources", str(sources), "--labels", str(labels),
         "--out", str(tmp_path / "out"), "--dry-run",
         "--min-languages", "0", "--min-test-rows", "0",
         "--min-labels-scored", "0", *extra],
        capture_output=True, text=True)
    return proc.returncode, proc.stderr


def _ledger(labels_trained, labels, removed=(), added=()):
    return {"version": 1,
            "baseline": {"corpus": "test", "labels_trained": labels_trained,
                         "labels": list(labels)},
            "removed": list(removed), "added": list(added)}


# The fixture declares no entry point, so skill_id_from_repo falls back to
# "<repo_name>.openvoiceos". The id is read, never guessed from the name.
ID = "ovos-skill-control.openvoiceos"


def test_e2e_a_pending_addition_does_not_fail_the_build(tmp_path):
    """The finding on #281, as a build.

    Baseline {alpha, beta}. The ledger folds alpha into beta and lists gamma
    as an addition. The pins carry the fold and do not carry gamma yet, so
    the corpus trains one label. Before this round the floor was 2 - 1 + 1 =
    2 and the build exited 1, blaming "a template the expansion drops" for a
    template nobody dropped.
    """
    rc, err = _run_build(
        tmp_path, {"beta": "say beta"},
        _ledger(2, [f"{ID}:alpha", f"{ID}:beta"],
                removed=[{"label": f"{ID}:alpha", "reason": "folded into beta",
                          "merged_into": f"{ID}:beta"}],
                added=[{"label": f"{ID}:gamma", "reason": "a new intent"}]))
    assert rc == 0, err


def test_e2e_an_unlisted_loss_still_fails_the_build(tmp_path):
    """The guard the fix must not weaken."""
    rc, err = _run_build(
        tmp_path, {"beta": "say beta"},
        _ledger(2, [f"{ID}:alpha", f"{ID}:beta"]))
    assert rc == 1
    assert f"{ID}:alpha" in err


def test_e2e_a_min_labels_below_the_floor_is_refused(tmp_path):
    """The documented refusal, which no test covered before this round."""
    rc, err = _run_build(
        tmp_path, {"alpha": "say alpha", "beta": "say beta"},
        _ledger(2, [f"{ID}:alpha", f"{ID}:beta"]),
        extra=("--min-labels", "1"))
    assert rc == 1
    assert "--min-labels 1 is below the floor 2" in err
    assert "not by typing a smaller number" in err


def test_e2e_a_min_labels_above_the_floor_is_allowed_and_binds(tmp_path):
    """The override raises the floor: the control that the refusal is not
    simply rejecting every --min-labels it is given."""
    rc, err = _run_build(
        tmp_path, {"alpha": "say alpha", "beta": "say beta"},
        _ledger(2, [f"{ID}:alpha", f"{ID}:beta"]),
        extra=("--min-labels", "3"))
    assert rc == 1
    assert "the corpus shrank to 2 labels, floor 3" in err
    assert "is below the floor" not in err


def test_the_count_gate_can_no_longer_fire_on_its_own(tmp_path):
    """The property that makes the derived floor safe to tighten.

    trained = expected - unlisted losses + unlisted gains, so trained can only
    fall below the floor when there are more unlisted losses than unlisted
    gains. Every such state fails the by-name check too. The count is a
    backstop behind the name check, never a verdict of its own, so no re-pin
    can be failed by the count while every loss in it is written down.
    """
    ledger = bfs.load_label_ledger(write(tmp_path, uneven()))
    baseline = {"s:a", "s:b", "s:c"}
    for bits in range(1 << 4):
        trained = set()
        for i, label in enumerate(sorted(baseline)):
            if bits >> i & 1:
                trained.add(label)
        if bits >> 3 & 1:
            trained.add("s:d")
        out = bfs.check_label_ledger(ledger, trained)
        if len(trained) < out["min_labels_derived"]:
            assert out["labels_lost_and_unlisted"], (
                f"{sorted(trained)} is below the floor "
                f"{out['min_labels_derived']} with every loss listed")
