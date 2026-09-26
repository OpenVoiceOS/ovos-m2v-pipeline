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


def test_a_slot_value_carrying_alternation_is_expanded_not_embedded():
    # A value pulled from an .entity file can itself carry template syntax.
    # Filling it with re.sub and never expanding again ships the syntax
    # verbatim as a training row; the fix must expand the filled result.
    value = "(lys|mørk|bleg|dyb|klar) (rød|orange|gul|grøn|blå|lilla|pink|brun|grå)"
    out = sentences("Vis mig farve {color}", hints={"color": [value]})
    assert sorted(out) == sorted(
        f"Vis mig farve {shade} {hue}"
        for shade in ["lys", "mørk", "bleg", "dyb", "klar"]
        for hue in ["rød", "orange", "gul", "grøn", "blå", "lilla", "pink",
                    "brun", "grå"])
    assert all("(" not in s and ")" not in s for s in out)


def test_a_slot_value_with_no_template_syntax_is_unchanged():
    out = sentences("play {track}", hints={"track": ["hey jude", "let it be"]})
    assert sorted(out) == ["play hey jude", "play let it be"]


def test_an_unfilled_slot_still_ships():
    assert sentences("set {alertkind} for {duration}",
                    hints={"alertkind": ["alarm", "timer"]}) == [
        "set alarm for {duration}", "set timer for {duration}"]


def test_the_post_build_check_counts_leftover_template_syntax():
    rows_dirty = [
        {"utterance": "Vis mig farve (lys|mørk) rød"},
        {"utterance": "play hey jude"},
    ]
    rows_clean = [
        {"utterance": "play hey jude"},
        {"utterance": "what type is the pokemon {pokemon}"},
    ]
    assert bfs.count_leftover_template_syntax(rows_dirty) == 1
    assert bfs.count_leftover_template_syntax(rows_clean) == 0


def test_a_bare_alternation_in_a_row_is_counted_not_missed():
    """The regression the manifest reported as zero.

    `complet|ple|plena` is an alternation whose parentheses the resource
    author left off. The detector matched `(a|b)` only, so a corpus holding
    715 of these rows reported a clean build. A fixture row in this shape
    keeps a zero from that counter honest.
    """
    rows = [
        {"utterance": "baixa la brillantor al complet|ple|plena per cent"},
        {"utterance": "muss ich mit Schneeansammlungen|Schneeverwehungen rechnen"},
        {"utterance": "Se espera nieve en el pronostico|"},
        {"utterance": "play hey jude"},
        {"utterance": "what type is the pokemon {pokemon}"},
    ]
    assert bfs.count_leftover_template_syntax(rows) == 3


def test_a_slot_value_carrying_a_bare_alternation_is_split_not_embedded():
    # The ca-ES brightness entity: three words for one setting, on one line,
    # with no parentheses. Substituted whole it ships the pipe as text.
    assert bfs.split_bare_alternation("complet|ple|plena") == [
        "complet", "ple", "plena"]
    out = sentences("baixa la brillantor al {brightness}",
                    hints={"brightness": ["complet", "ple", "plena"]})
    assert sorted(out) == [
        "baixa la brillantor al complet",
        "baixa la brillantor al ple",
        "baixa la brillantor al plena",
    ]
    assert all("|" not in s for s in out)


def test_a_value_whose_alternation_is_grouped_is_left_to_the_expander():
    value = "(lys|mørk) rød"
    assert bfs.split_bare_alternation(value) == [value]


def test_a_value_with_no_alternation_is_one_value():
    assert bfs.split_bare_alternation("hey jude") == ["hey jude"]


def test_a_template_with_a_bare_alternation_has_no_defined_scope():
    # A value has one possible reading; a template line has several, so the
    # line is dropped rather than expanded to a guess.
    assert bfs.has_bare_alternation("enfosqueix|atenua una mica")
    assert bfs.has_bare_alternation("Se espera nieve en el pronostico|")
    assert not bfs.has_bare_alternation("(baixa|aumenta) la brillantor")
    assert not bfs.has_bare_alternation("play {track}")
    # An optional segment is grammar, not a defect: `expand` reads `[the]`
    # and returns both readings. Reading it as one drops sound templates and
    # takes their labels with them.
    assert not bfs.has_bare_alternation("play [the] music")
    assert not bfs.has_bare_alternation("play [the] (a|b) music")


def test_the_unfilled_slot_counter_reads_the_finished_corpus():
    rows = [
        {"utterance": "what type is the pokemon {pokemon}"},
        {"utterance": "set alarm for {duration}"},
        {"utterance": "play hey jude"},
    ]
    assert bfs.count_rows_with_an_unfilled_slot(rows) == 2


def test_ambiguity_is_counted_apart_from_duplication():
    """The same sentence under two labels is a conflict, not a duplicate."""
    rows = [
        {"lang": "en-US", "utterance": "stop", "label": "a:stop"},
        {"lang": "en-US", "utterance": "stop", "label": "b:halt"},
        {"lang": "en-US", "utterance": "play hey jude", "label": "c:play"},
        {"lang": "de-DE", "utterance": "stop", "label": "a:stop"},
    ]
    assert bfs.count_ambiguous_rows(rows) == 2
