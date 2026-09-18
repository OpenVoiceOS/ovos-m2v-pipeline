#!/usr/bin/env python3
"""Build the intent corpus from the skills themselves.

The skills are the data. Every locale resource a running skill would load is
the training side, and the gold sentences the fleet already ships are the test
side. Provenance alone does not keep the two disjoint: a gold sentence and an
expansion of that skill's own template are both somebody reaching for the
obvious phrasing of the same intent, and they collide constantly regardless
of which file either one was read from. So after both sides are built, every
training row whose utterance matches a gold utterance is removed from the
training side. The comparison is string equality after lowercasing,
collapsing whitespace, and stripping leading and trailing punctuation --
the same trailing-punctuation fold ``ovos-utterance-normalizer`` applies to
an incoming utterance before an intent engine ever sees it, so "play rock?"
in gold and "play rock" in a template expansion are one string here because
they are one string at match time. It is still not accent-insensitive and
not semantic. The gold side is never thinned: dropping the gold rows a
skill author happened to also think of would leave the phrasings nobody
anticipated, a biased sample rather than a smaller one. The manifest
records the measured overlap between the two sides after the removal under
both the punctuation-insensitive comparison the removal itself uses and the
narrower exact-string comparison, both of which must be zero, and how many
templates the removal left with no rows at all, because that count is the
price paid for the fix and a manifest that reports only the zero is how
this defect gets rebuilt. The test side itself is a claim that must hold a
floor too: a gold glob that resolves to nothing builds a corpus with zero
test rows and reports the overlap as a vacuous zero, so ``--min-test-rows``
and ``--min-labels-scored`` gate the test side exactly as ``--min-labels``
and ``--min-languages`` gate the training side.

Three resource kinds, each read for what it is. An ``.intent`` file carries
templates. An ``.entity`` file carries example values for a slot -- a hint per
OVOS-INTENT-3 5.2, never a closed set, because a value outside it must stay
matchable. A slot with no examples still produces a row: INTENT-1 5.2 makes a
slot free-form capture and 5.4 says it fills without a value set, so the
placeholder is left literal exactly as the runtime leaves it. A ``.voc`` file carries the keywords that trigger an intent, and is
never read as slot values: ``repeat.voc`` holding "every" and "frequency" says
how a user asks for recurrence, not what a ``{repeat}`` slot contains.

Language comes from the locale directory a file sits in and stays with the
row, so a value attested only under ``en-US`` fills only English templates.
Dialect directories keep their tags; collapsing them is a publication step
applied to the finished dataset. OVOS-INTENT-2 2 makes tag comparison
case-insensitive ("`en-us` and `en-US` denote the same language"), so a
directory's case is not part of a language's identity: ``fr-fr`` and
``fr-FR`` fold onto one canonical spelling before a row is built. That is
the only equivalence this builder applies -- ``eu``/``eu-ES``,
``cmn-CN``/``zh-CN`` and similar macrolanguage or variant pairs differ by
more than case and stay distinct.
"""
import argparse
import collections
import json
import re
import string
import subprocess
import sys
from pathlib import Path

import yaml

from ovos_spec_tools.expansion import expand
from ovos_spec_tools.lint import declared_slot_types

import typed_slots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill_labels  # noqa: E402
from build_eval import read_gold  # noqa: E402  (the one gold reader)

#: A locale directory: an ISO subtag, optionally with a script and a region.
LOCALE_DIR = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z]{4})?(?:-(?:[A-Za-z]{2}|\d{3}))?$")

#: Where a resource sits, whatever nests the locale root.
RESOURCE = re.compile(r"(?:^|.*/)locale/([^/]+)/(?:.*/)?([^/]+)\.(intent|entity|voc)$")

#: Gold lives beside the end-to-end suite that reads it.
GOLD = re.compile(r"(?:^|.*/)golden_utterances(?:_([A-Za-z-]+))?\.jsonl$")

#: OVOS-INTENT-2 2: "A resource base name MUST consist only of lowercase
#: ASCII letters, digits, and underscores, and MUST NOT contain whitespace".
COMPLIANT_BASE_NAME = re.compile(r"^[a-z0-9_]+$")

#: A template producing more than this is a generator, not a phrasing.
EXPANSION_CAP = 2000


def strip_groups(text: str) -> str:
    """*text* with every balanced `(...)` and `[...]` group removed.

    Both are grammar: `(a|b)` is an alternation and `[a]` an optional
    segment, and the expander reads each correctly. What is left is the part
    of a line the grammar does not account for, so a `|` still in it belongs
    to nothing: the author wrote an alternation and left off its
    parentheses.
    """
    out = []
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def has_bare_alternation(text: str) -> bool:
    """A `|` none of the grammar's own groups covers.

    Only the pipe. An optional segment is legal template syntax that expands,
    so reading `[the]` as a defect drops thousands of sound templates and
    takes their labels with them -- the build's own label floor catches that,
    which is how this check was caught being too wide.
    """
    return "|" in strip_groups(text)


def split_bare_alternation(value: str) -> list:
    """The alternatives a resource value names, splitting a bare `|`.

    A value is one whole thing, so a `|` in it that no group covers separates
    two values the author wrote on one line -- `complet|ple|plena` is three
    Catalan words for one brightness setting. Substituted whole, it ships as
    the literal string `complet|ple|plena` in a training row, which teaches a
    model to say the pipe out loud. The scope is unambiguous here, unlike the
    same mistake in a template, so the value is split rather than dropped.
    """
    if not has_bare_alternation(value) or "|" not in strip_groups(value):
        return [value]
    parts, depth, current = [], 0, []
    for ch in value:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "|" and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout


def tree(repo: Path, rev: str):
    return git(repo, "ls-tree", "-r", "--name-only", rev).split("\n")


def show(repo: Path, rev: str, path: str) -> str:
    try:
        return git(repo, "show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return ""


def normalize_utterance(text: str) -> str:
    """Lowercase and collapse whitespace, the narrower overlap comparison.

    Exact string match after this normalisation, nothing wider: not
    accent-insensitive, not semantic. Kept and reported alongside the
    punctuation-insensitive comparison below because the two answer
    different questions, and collapsing them into one number would hide
    which normalisation the removal actually rests on.
    """
    return re.sub(r"\s+", " ", text.strip()).lower()


def normalize_utterance_punct_insensitive(text: str) -> str:
    """As ``normalize_utterance``, plus stripping leading/trailing punctuation.

    ``ovos-utterance-normalizer``, the runtime transformer every incoming
    utterance passes through before an intent engine matches it, strips
    edge punctuation with ``utterance.strip(string.punctuation).strip()``.
    A gold row ending in "?" and a training expansion without one are the
    same string at that point, so the removal this builder performs is
    matched on this wider equivalence, not the narrower one above. Still
    not accent-insensitive, not semantic, and inner punctuation is left
    alone -- only what the runtime strips is stripped here.
    """
    return normalize_utterance(text).strip(string.punctuation).strip()


def normalize_lang(tag: str) -> str:
    """Canonical spelling of a BCP-47 tag, per OVOS-INTENT-2 2.

    Tag comparison is case-insensitive, so a locale directory's case is not
    part of a language's identity. Lowercase the primary and extended
    subtags, titlecase a 4-letter script, uppercase a 2-letter region; a
    3-digit region is a number and has no case to fold.
    """
    parts = tag.split("-")
    out = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            out.append(part.title())
        elif len(part) == 2 and part.isalpha():
            out.append(part.upper())
        else:
            out.append(part)
    return "-".join(out)


def read_skill(repo: Path, rev: str, repo_name: str, stats: collections.Counter,
               violations: set, dropped_templates: list = None):
    """Templates, hints and keywords a skill ships, per language."""
    if dropped_templates is None:
        dropped_templates = []
    templates = []                                   # (lang, label, template)
    hints = collections.defaultdict(dict)            # lang -> name -> values
    keywords = collections.defaultdict(dict)         # lang -> name -> words
    # the one label function (skill_labels): the id the entry point declares
    skill_id = skill_labels.skill_id_from_repo(repo, rev, repo_name)
    if not skill_labels.declares_skill_id(repo, rev):
        stats["skill_id_assumed_from_repo_name"] += 1
    for path in tree(repo, rev):
        if not path or path.split("/")[0] in {"test", "tests"}:
            continue
        m = RESOURCE.match(path)
        if not m:
            continue
        lang, name, kind = m.group(1), m.group(2), m.group(3)
        if not LOCALE_DIR.match(lang):
            stats["resource_outside_a_locale"] += 1
            continue
        lang = normalize_lang(lang)
        if not COMPLIANT_BASE_NAME.match(name):
            # The corpus still reads it -- the fleet is what it is -- but a
            # base name carrying a dot, a dash or a capital breaks the rule
            # the loader relies on, and the fallback that rescues its gold
            # rows would otherwise hide it.
            violations.add(f"{repo_name}: {name}.{kind}")
        body = show(repo, rev, path)
        values = [l.strip() for l in body.splitlines()
                  if l.strip() and not l.strip().startswith("#")]
        if kind == "intent":
            label = skill_labels.label(skill_id, name)
            for line in values:
                # A `|` the grammar's groups do not cover binds nothing the
                # grammar defines. Where the same mistake in a resource value
                # has one possible reading, here it has several -- the author
                # may have meant two words, two phrases or the whole line --
                # so the line is dropped and counted rather than expanded to
                # a guess. `Se espera nieve en el pronostico|` is the shape:
                # nobody can say what the trailing pipe was for.
                if has_bare_alternation(line):
                    stats["template_with_a_bare_alternation_dropped"] += 1
                    dropped_templates.append(f"{repo_name} {lang}: {line}")
                    continue
                templates.append((lang, label, line))
        elif kind == "entity":
            # Two directories differing only in case are one language after
            # normalisation, so their entity files are the same file under
            # two spellings: union the values rather than letting whichever
            # sorts last silently discard the other's.
            existing = hints[lang].setdefault(name.lower(), [])
            for value in values:
                alternatives = split_bare_alternation(value)
                if len(alternatives) > 1:
                    stats["resource_value_bare_alternation_split"] += 1
                existing.extend(v for v in alternatives if v not in existing)
        else:
            existing = keywords[lang].setdefault(name.lower(), [])
            for value in values:
                alternatives = split_bare_alternation(value)
                if len(alternatives) > 1:
                    stats["resource_value_bare_alternation_split"] += 1
                existing.extend(v for v in alternatives if v not in existing)
    return templates, hints, keywords


def fill(template: str, hints: dict, keywords: dict, stats: collections.Counter,
         lang: str = "en-US"):
    """Sentences a template produces, with its own language's resources.

    `<name>` references a keyword file, which is how a template says "any of
    these words here". `{name}` is a slot: filled from the language's hints
    when it has them, and left unfilled otherwise -- an unfilled slot is a
    phrasing this corpus cannot use, not a phrasing the skill cannot match.
    """
    # `expand` strips a type prefix, so `{number:offset}` reaches the loop
    # below as `{offset}` and the type is only knowable from the template
    # itself (OVOS-INTENT-4 6.1).
    try:
        slot_types = declared_slot_types([template]) or {}
    except Exception:
        slot_types = {}
    try:
        sentences = expand(template, keywords)
    except Exception:
        stats["template_would_not_expand"] += 1
        return []
    if len(sentences) > EXPANSION_CAP:
        stats["template_over_cap"] += 1
        sentences = sentences[:EXPANSION_CAP]
    out = []
    for s in sentences:
        slots = {n.lower() for n in re.findall(r"\{([^}]+)\}", s)}
        if not slots:
            out.append(s)
            continue
        # A slot with no examples is not a dead template. INTENT-1 5.2 makes
        # a slot free-form capture, 5.4 says a slot with no value set still
        # fills, and the runtime agrees: `expand_entities` passes a sample
        # through with the placeholder left literal and embeds it that way.
        # The corpus mirrors the runtime rather than discarding the phrasing.
        # A typed slot is filled from the parser for its type in THIS
        # language, never from an .entity and never left as a brace: the
        # runtime binds it by asking that same parser, so a row carrying the
        # placeholder teaches a surface the runtime never produces. A type
        # with no generator for this language drops the sentence rather than
        # filling it with another language's words.
        typed_here = {n: slot_types[n] for n in slots if n in slot_types}
        if typed_here:
            values = {n: typed_slots.values_for(t, lang)
                      for n, t in typed_here.items()}
            if not all(values.values()):
                stats["sentence_dropped_no_typed_values_for_this_language"] += 1
                continue
            partials = [s]
            for name, produced in values.items():
                grown = []
                for partial in partials:
                    for value in produced:
                        grown.append(re.sub(r"\{" + re.escape(name) + r"\}", value,
                                            partial, flags=re.IGNORECASE))
                partials = grown[:EXPANSION_CAP]
            stats["sentences_from_a_typed_slot"] += len(partials)
            slots = slots - set(typed_here)
            if not slots:
                out.extend(partials)
                continue
            # an untyped slot is left to the hint pass below, per sentence
            for partial in partials:
                out.extend(fill(partial, hints, keywords, stats, lang))
            continue

        if any(n not in hints for n in slots):
            stats["sentences_kept_with_an_unfilled_slot_before_filling"] += 1
        filled = [s]
        for slot in sorted(n for n in slots if n in hints):
            nxt = []
            for partial in filled:
                for value in hints[slot]:
                    nxt.append(re.sub(r"\{" + re.escape(slot) + r"\}", value,
                                      partial, flags=re.IGNORECASE))
            filled = nxt[:EXPANSION_CAP]
        # A hint value is substituted verbatim, and a value can itself carry
        # template syntax (an `.entity` line copied from a template, or one
        # written with alternation in it). Left alone, that syntax ships as a
        # literal training row instead of the sentences it denotes, so every
        # filled string is expanded again. `expand` is eager and raises
        # rather than returning a partial sample set, so a fill that produces
        # a malformed string is dropped whole, the same way a malformed
        # template is dropped above -- never partially kept.
        reexpanded = []
        for f in filled:
            try:
                grown = expand(f, {})
            except Exception:
                stats["filled_sentence_would_not_expand"] += 1
                continue
            if len(grown) > 1:
                stats["rows_expanded_after_filling"] += len(grown)
            reexpanded.extend(grown)
        if len(reexpanded) > EXPANSION_CAP:
            stats["sentence_over_cap_after_fill"] += 1
            reexpanded = reexpanded[:EXPANSION_CAP]
        out.extend(reexpanded)
    return out


def count_leftover_template_syntax(rows) -> int:
    """Written rows that still carry template syntax other than `{slot}`.

    A slot left unfilled (INTENT-1 5.4) is the documented, correct shape of
    a row -- `{slot}` is excluded. Anything else the grammar defines is never
    a valid training row.

    This counted `(a|b)` and `[a]` only, so it read 0 on a corpus holding 715
    rows whose alternation had lost its parentheses -- `complet|ple|plena`,
    the exact shape the expander leaves behind. A detector that cannot see
    the defect it exists for is worse than no detector: it reports a clean
    build. A bare `|`, `(`, `)`, `[` or `]` all count now, so a zero here has
    to be earned.
    """
    leftover = re.compile(r"[()\[\]|]")
    return sum(1 for row in rows if leftover.search(row["utterance"]))


def count_rows_with_an_unfilled_slot(rows) -> int:
    """Written rows that still carry a `{slot}`.

    Read off the finished corpus, not counted as the expander goes. The
    per-stage counter says how many SENTENCES were kept with an unfilled
    slot, before filling multiplied them and before dedup and the gold
    overlap removed some, so it is not the number of rows that ship and must
    not be read as one.
    """
    return sum(1 for row in rows if re.search(r"\{[^}]+\}", row["utterance"]))


def count_ambiguous_rows(rows) -> int:
    """Rows whose locale and utterance carry a label another row disagrees on.

    Not duplicates. A duplicate is the same sentence for the same label and
    says nothing new; these are the same sentence for DIFFERENT labels, which
    is a conflict the model is asked to resolve and cannot. They are counted
    and named separately because folding them into a duplicate count hides a
    labelling problem behind a housekeeping figure.
    """
    labels = collections.defaultdict(set)
    for row in rows:
        labels[(row["lang"], row["utterance"])].add(row["label"])
    conflicted = {key for key, names in labels.items() if len(names) > 1}
    return sum(1 for row in rows
               if (row["lang"], row["utterance"]) in conflicted)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", default="train/sources.yaml")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    # Floors rather than an equality: the corpus grows as skills gain
    # resources, and it must never quietly shrink. Both numbers move the
    # moment the expansion step starts discarding templates again, which is
    # the failure this build already had once and which no count revealed.
    # Measured at the pins in train/sources.yaml: 235 labels trained (dev
    # measured 227). Folds remove labels: date-time, randomness, wallpapers
    # and pokepedia renamed their resources to underscored base names,
    # volume and weather consolidated several intents into slot intents,
    # and the it-IT count file lost its capital. The new pokepedia pin and
    # the skill_refs that were missing add more labels than the folds
    # remove, so the net count goes up and the floor goes up with it.
    # 233 after the OVOS-INTENT-2 2 rename wave re-pin: ovos-skill-alerts#216
    # merged DeleteTodoEntries and QueryTodoEntries into their list-kind
    # siblings, so two labels leave and every other rename is one to one
    # (measured: 235 labels at the old pins, 233 at the new, 29 removed,
    # 27 added).
    # 232 after the gold-skill pin move: ovos-skill-moviemaster#79 dropped the
    # dead movie_information and movie_production intents on purpose.
    ap.add_argument("--min-labels", type=int, default=232)
    # Measured on the same tree: 53 distinct locale directories folded to 52
    # once fa-ir/fa-IR merged under OVOS-INTENT-2 2's case-insensitive tag
    # comparison. The prior floor of 53 counted that pair twice.
    ap.add_argument("--min-languages", type=int, default=52)
    # The zero this build asserts on the overlap is satisfied trivially when
    # the test side is empty: a gold glob renamed or moved out from under the
    # builder still builds rc=0 with test_rows=0 and every overlap zero by
    # construction, scoring nothing while reporting the cleanest possible
    # number. Floors on the test side close that, the same way min-labels
    # and min-languages close it on the training side. Measured against a
    # real build over dev@72d73a5 (`--workspace ~/AgentWorkspaces`, shipped
    # source pins): test_rows=1412.
    # Measured after every gold PR of the v6.1 wave merged: test_rows=2380.
    ap.add_argument("--min-test-rows", type=int, default=2380)
    # Measured on the same build: labels_scored=164.
    # Measured on the same build: labels_scored=191.
    ap.add_argument("--min-labels-scored", type=int, default=191)
    # A base name that breaks OVOS-INTENT-2 2 is read anyway, because the
    # corpus must build against the fleet as it stands. Pinning the count
    # stops the set growing while the rename campaign brings it down; a zero
    # here would be red on arrival and loosened away by whoever ran it next.
    ap.add_argument("--max-noncompliant-base-names", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.sources).read_text(encoding="utf-8"))
    ws = Path(args.workspace or cfg["workspace"]).expanduser()
    refs = cfg["skill_refs"]["refs"]

    stats = collections.Counter()
    train, test = [], []
    labels_trained, labels_scored = set(), set()
    labels_with_templates = set()
    no_gold, no_resources = [], []
    violations = set()
    dropped_templates = []

    for key, rev in sorted(refs.items()):
        repo, repo_name = ws / key, key.rsplit("/", 1)[-1]
        if not repo.is_dir():
            stats["skill_missing_clone"] += 1
            continue
        templates, hints, keywords = read_skill(repo, rev, repo_name, stats,
                                                violations, dropped_templates)
        if not templates:
            no_resources.append(repo_name)
        for _, label, _ in templates:
            labels_with_templates.add(label)
        for lang, label, template in templates:
            for sentence in fill(template, hints.get(lang, {}),
                                 keywords.get(lang, {}), stats, lang):
                train.append({"lang": lang, "label": label,
                              "utterance": sentence, "source": f"skill:{repo_name}",
                              "template": template})
                labels_trained.add(label)
        # The gold side is read by build_eval.read_gold, the same reader the
        # eval builder uses, and labelled by the same function as the train
        # rows above. No spelling fallback: a gold label the train side does
        # not carry is reported by the manifest and refused by the census.
        gold = read_gold(repo, rev, repo_name, stats)
        if not gold:
            no_gold.append(repo_name)
        for row in gold:
            test.append(row)
            labels_scored.add(row["label"])

    seen = set()
    deduped = []
    for row in train:
        key = (row["label"], row["lang"], row["utterance"].lower())
        if key in seen:
            stats["train_duplicate_rows_removed"] += 1
            continue
        seen.add(key)
        deduped.append(row)
    train = deduped

    # A gold sentence and an expansion of the same skill's own template are
    # both somebody reaching for the obvious phrasing of the same intent, so
    # they collide regardless of which file either was read from. Every
    # training row that collides with a gold row -- match after lowercasing,
    # whitespace collapse, and stripping edge punctuation, the same fold the
    # runtime utterance normalizer applies before an intent engine matches
    # -- is removed from the training side; the gold side is never touched.
    # Measured, not assumed: the overlap under both comparisons and the
    # templates the removal silences all go in the manifest.
    gold_normalized = {normalize_utterance(r["utterance"]) for r in test}
    gold_normalized_wide = {
        normalize_utterance_punct_insensitive(r["utterance"]) for r in test}
    templates_before = collections.defaultdict(set)
    for row in train:
        templates_before[row["template"]].add(row["label"])
    train_gold_overlap = [
        row for row in train
        if normalize_utterance_punct_insensitive(row["utterance"])
        in gold_normalized_wide]
    train = [row for row in train
             if normalize_utterance_punct_insensitive(row["utterance"])
             not in gold_normalized_wide]
    templates_after = {row["template"] for row in train}
    silenced_templates = sorted(t for t in templates_before
                                if t not in templates_after)
    silenced_labels = sorted({label for t in silenced_templates
                              for label in templates_before[t]})
    # The overlap this fix removes may have been the only training rows a
    # label had; re-derive labels_trained from what survives rather than
    # from the pre-removal pass, or the manifest would call a label trained
    # when its last row was just deleted.
    labels_trained = {row["label"] for row in train}
    train_gold_overlap_after = sum(
        1 for row in train if normalize_utterance(row["utterance"]) in gold_normalized)
    train_gold_overlap_after_punct_insensitive = sum(
        1 for row in train
        if normalize_utterance_punct_insensitive(row["utterance"])
        in gold_normalized_wide)

    # A gold sentence naming a label no skill trains is two different
    # findings wearing one shape, and they ask for opposite work. If the
    # skill ships the intent and it produces no rows, the intent is starved
    # of slot examples and the skill is fine. If the skill does not ship the
    # intent at all, the gold row asserts a routing its own skill never
    # supported -- either written wrong, or written before a rename nobody
    # carried into the gold file.
    scored_labels = {r["label"] for r in test}
    starved = sorted(scored_labels & labels_with_templates - labels_trained)
    unsupported = sorted(scored_labels - labels_with_templates)

    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "labels_trained": len(labels_trained),
        "labels_scored": len(labels_scored & labels_trained),
        "labels_never_scored": len(labels_trained - labels_scored),
        "languages_train": len({r["lang"] for r in train}),
        "languages_test": len({r["lang"] for r in test}),
        "train_gold_overlap_removed": len(train_gold_overlap),
        "train_gold_overlap_after_fix": train_gold_overlap_after,
        "train_gold_overlap_after_fix_punct_insensitive":
            train_gold_overlap_after_punct_insensitive,
        "templates_silenced": len(silenced_templates),
        "templates_silenced_detail": silenced_templates,
        "labels_with_a_silenced_template": silenced_labels,
        # A row still carrying template syntax other than an unfilled `{slot}`
        # is unusable, whatever the per-stage counters above say. This is the
        # number that would have caught the shipped-alternation defect: it
        # reads the finished corpus rather than trusting the builder's own
        # account of its work.
        "train_rows_with_leftover_template_syntax":
            count_leftover_template_syntax(train),
        # Both read the finished corpus. The per-stage counters under "stats"
        # count sentences as the expander sees them, which is a different
        # number and carries a different name.
        "train_rows_with_an_unfilled_slot":
            count_rows_with_an_unfilled_slot(train),
        "train_rows_ambiguous_same_utterance_different_label":
            count_ambiguous_rows(train),
        "templates_dropped_for_a_bare_alternation": len(dropped_templates),
        "templates_dropped_for_a_bare_alternation_detail":
            sorted(dropped_templates),
        "gold_flags_are_not_evidence": (
            "every gold sentence in this fleet was written by a model, so the "
            "machine_generated field is unreliable wherever it claims False "
            "and is not read; needs_manual marks a row somebody meant a human "
            "to check, and whether that check happened is unrecorded"),
        "skills_without_gold": sorted(no_gold),
        "skills_without_locale_resources": sorted(no_resources),
        "noncompliant_base_names": len(violations),
        "noncompliant_base_names_detail": sorted(violations),
        "label_function": "train/skill_labels.py (no spelling fallback)",
        "gold_labels_starved_of_slot_examples": starved,
        "gold_labels_no_skill_supports": unsupported,
        "stats": dict(stats),
    }
    print(json.dumps({k: v for k, v in report.items()
                      if not isinstance(v, list)}, indent=2))
    short = []
    if report["labels_trained"] < args.min_labels:
        short.append(
            f"the corpus shrank to {report['labels_trained']} labels, floor "
            f"{args.min_labels}: a template the expansion drops takes its "
            f"label with it and no other count shows the loss")
    if report["languages_train"] < args.min_languages:
        short.append(
            f"the corpus shrank to {report['languages_train']} languages, "
            f"floor {args.min_languages}")
    if report["test_rows"] < args.min_test_rows:
        short.append(
            f"the gold side shrank to {report['test_rows']} test rows, "
            f"floor {args.min_test_rows}: a moved or renamed gold glob "
            f"that matches nothing builds an overlap of zero by having "
            f"nothing to overlap with, and no other count shows the loss")
    if report["labels_scored"] < args.min_labels_scored:
        short.append(
            f"the gold side scores only {report['labels_scored']} labels, "
            f"floor {args.min_labels_scored}")
    if report["train_gold_overlap_after_fix"] != 0:
        short.append(
            f"{report['train_gold_overlap_after_fix']} training rows still "
            f"match a gold utterance after the removal pass: the fix did "
            f"not do what it claims")
    if report["train_gold_overlap_after_fix_punct_insensitive"] != 0:
        short.append(
            f"{report['train_gold_overlap_after_fix_punct_insensitive']} "
            f"training rows still match a gold utterance once edge "
            f"punctuation is stripped, the fold the runtime normalizer "
            f"applies before an intent engine ever compares the two: the "
            f"fix did not do what it claims")
    cap = args.max_noncompliant_base_names
    if cap is not None and report["noncompliant_base_names"] > cap:
        short.append(
            f"{report['noncompliant_base_names']} resource base names break "
            f"OVOS-INTENT-2 2, ceiling {cap}: the set may shrink as skills "
            f"are renamed and must never gain a member")
    if short:
        for line in short:
            print(f"[gate] {line}", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("test.jsonl", test)):
        with (out / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
