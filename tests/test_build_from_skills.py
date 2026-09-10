"""A slot with no examples must still produce a row.

INTENT-1 §5.2 makes a slot free-form capture and §5.4 says a slot with no
value set still fills, and the runtime agrees: `expand_entities` passes a
sample through with its placeholder left literal and the pipeline embeds that
string as the intent's prototype. A corpus that discards those templates
reproduces something no runtime does, and it does it silently -- the rows
simply are not there, so no count reveals the loss.
"""
import collections
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ovos_spec_tools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_from_skills as bfs  # noqa: E402


def sentences(template, hints=None, keywords=None):
    import collections
    return bfs.fill(template, hints or {}, keywords or {}, collections.Counter())


def test_a_slot_with_no_examples_keeps_its_marker():
    assert sentences("what type is the pokemon {pokemon}") == [
        "what type is the pokemon {pokemon}"]


def test_a_slot_with_examples_is_filled_with_each_of_them():
    out = sentences("play {track}", hints={"track": ["hey jude", "let it be"]})
    assert sorted(out) == ["play hey jude", "play let it be"]


def test_one_filled_slot_beside_one_unfilled_one():
    out = sentences("set {alertkind} for {duration}",
                    hints={"alertkind": ["alarm", "timer"]})
    assert sorted(out) == ["set alarm for {duration}",
                           "set timer for {duration}"]


def test_a_keyword_reference_expands_to_its_words():
    out = sentences("<greeting> world", keywords={"greeting": ["hello", "hi"]})
    assert sorted(out) == ["hello world", "hi world"]


def test_a_keyword_file_is_never_read_as_slot_values():
    # `repeat.voc` holds the words that ask for recurrence -- "every",
    # "frequency" -- and filling `{repeat}` from them would teach the model
    # that a repeat slot contains the word "frequency".
    out = sentences("repeat {repeat}", keywords={"repeat": ["every", "frequency"]})
    assert out == ["repeat {repeat}"]


def test_the_expansion_is_bounded():
    hints = {"n": [str(i) for i in range(bfs.EXPANSION_CAP * 2)]}
    assert len(sentences("count to {n}", hints=hints)) <= bfs.EXPANSION_CAP


def test_a_dotted_gold_label_resolves_to_the_underscored_file():
    # moviemaster's gold writes `movie.description`; the file is
    # `movie_description`, and the skill's own suite bridges the two.
    assert bfs.resolve_gold_label(
        "movie.description", {"movie_description"}) == "movie_description"


def test_a_genuinely_dotted_filename_is_left_alone():
    # volume really does ship `volume.mute.intent`. Mapping its dots would
    # invent an intent the skill does not have, so the shipped name wins.
    shipped = {"volume.mute", "volume_mute"}
    assert bfs.resolve_gold_label("volume.mute", shipped) == "volume.mute"


def test_an_unknown_label_is_returned_unchanged_to_be_reported():
    assert bfs.resolve_gold_label("count_to_N", {"count_to_n"}) == "count_to_N"


def test_an_ambiguous_underscoring_refuses_to_resolve():
    # Two shipped intents that differ only in a dot fold onto one
    # underscored form. The gold label matches neither as written, and
    # picking the one that happens to equal the folded form scores the row
    # against a label nobody asserted. It resolves to neither.
    shipped = {"a_b_c", "a.b_c"}
    assert bfs.resolve_gold_label("a.b.c", shipped) == "a.b.c"


def test_an_unambiguous_underscoring_still_resolves():
    assert bfs.resolve_gold_label("a.b.c", {"a_b_c", "unrelated"}) == "a_b_c"


def _write_skill_repo(root, files):
    """A throwaway git repo with the given path -> content files, committed."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@test.invalid", "-c", "user.name=test",
         "commit", "-q", "-m", "skill"],
        cwd=root, check=True)
    return root


def test_two_skills_sharing_an_entity_name_do_not_share_its_values(tmp_path):
    alpha = _write_skill_repo(tmp_path / "ovos-skill-alpha", {
        "locale/en-US/pick.intent": "play {genre}\n",
        "locale/en-US/genre.entity": "rock\n",
    })
    beta = _write_skill_repo(tmp_path / "ovos-skill-beta", {
        "locale/en-US/pick.intent": "play {genre}\n",
        "locale/en-US/genre.entity": "schlager\n",
    })
    a_templates, a_hints, a_keywords = bfs.read_skill(
        alpha, "HEAD", "ovos-skill-alpha", collections.Counter(), set())
    b_templates, b_hints, b_keywords = bfs.read_skill(
        beta, "HEAD", "ovos-skill-beta", collections.Counter(), set())

    _, _, a_template = a_templates[0]
    _, _, b_template = b_templates[0]
    a_out = bfs.fill(a_template, a_hints["en-US"], a_keywords["en-US"],
                      collections.Counter())
    b_out = bfs.fill(b_template, b_hints["en-US"], b_keywords["en-US"],
                      collections.Counter())

    assert a_out == ["play rock"]
    assert b_out == ["play schlager"]


def test_two_locale_directories_differing_only_in_case_are_one_language(tmp_path):
    # OVOS-INTENT-2 2: "Tag comparison is case-insensitive: `en-us` and
    # `en-US` denote the same language." A skill that ships both directory
    # spellings must produce ONE language in the manifest, and rows from
    # either directory must carry the same `lang` value.
    delta = _write_skill_repo(tmp_path / "ovos-skill-delta", {
        "locale/fr-fr/pick.intent": "joue {genre}\n",
        "locale/fr-fr/genre.entity": "rock\n",
        "locale/fr-FR/pick.intent": "joue {genre}\n",
        "locale/fr-FR/genre.entity": "jazz\n",
    })
    templates, hints, keywords = bfs.read_skill(
        delta, "HEAD", "ovos-skill-delta", collections.Counter(), set())

    langs = {lang for lang, _, _ in templates}
    assert langs == {"fr-FR"}

    counter = collections.Counter()
    out = set()
    for lang, label, template in templates:
        out.update(bfs.fill(template, hints.get(lang, {}),
                            keywords.get(lang, {}), counter))
    assert out == {"joue rock", "joue jazz"}


def test_a_row_present_under_both_spellings_survives_dedup_once():
    train = [
        {"label": "x:pick", "lang": "fr-FR", "utterance": "joue rock",
         "source": "skill:delta", "template": "joue {genre}"},
        {"label": "x:pick", "lang": "fr-fr", "utterance": "joue rock",
         "source": "skill:delta", "template": "joue {genre}"},
    ]
    normalized = [{**row, "lang": bfs.normalize_lang(row["lang"])} for row in train]
    seen, deduped = set(), []
    for row in normalized:
        key = (row["label"], row["lang"], row["utterance"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    assert len(deduped) == 1
    assert deduped[0]["lang"] == "fr-FR"


def test_a_macrolanguage_and_its_member_stay_distinct():
    # `eu` (Basque, a macrolanguage-less ISO code here used as a bare
    # primary tag) and `eu-ES` differ by more than case -- no clause in
    # OVOS-INTENT-2 makes them equal -- so normalisation must not fold them.
    assert bfs.normalize_lang("eu") == "eu"
    assert bfs.normalize_lang("eu-ES") == "eu-ES"
    assert bfs.normalize_lang("eu") != bfs.normalize_lang("eu-ES")


def test_an_entity_is_scoped_to_its_own_language(tmp_path):
    gamma = _write_skill_repo(tmp_path / "ovos-skill-gamma", {
        "locale/en-US/pick.intent": "play {genre}\n",
        "locale/en-US/genre.entity": "rock\n",
        "locale/de-DE/pick.intent": "spiele {genre}\n",
        "locale/de-DE/genre.entity": "schlager\n",
    })
    templates, hints, keywords = bfs.read_skill(
        gamma, "HEAD", "ovos-skill-gamma", collections.Counter(), set())

    counter = collections.Counter()
    out = {
        lang: bfs.fill(template, hints.get(lang, {}), keywords.get(lang, {}),
                       counter)
        for lang, label, template in templates
    }

    assert out["en-US"] == ["play rock"]
    assert out["de-DE"] == ["spiele schlager"]
