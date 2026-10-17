"""Benchmark datasets — loaded from the Hugging Face Hub.

Two OpenVoiceOS evaluation datasets are supported, both shaped the same way:
a ``<lang>-templates`` config (training templates) and a ``<lang>-test`` config
(labelled evaluation utterances).

- ``intents-for-eval`` — ``OpenVoiceOS/intents-for-eval``. 50 intents, a test
  set split into ``template`` / ``paraphrase`` / ``near_ood`` / ``asr_noise`` /
  ``typos`` (labelled) and ``far_ood`` (no-match).
- ``massive`` — ``OpenVoiceOS/massive-templates``, an OVOS-templated rebuild of
  the MASSIVE intent corpus. ~60 intents, a single labelled test split, no
  no-match cases.

Every engine in this benchmark is a template / sample matcher, so it trains on
the ``-templates`` config and is evaluated on the ``-test`` config.

Slots are turned into entities: every ``{slot}`` placeholder carries example
values, collected into ``Bundle.entities`` so engines can register them (the
equivalent of a padatious ``.entity`` file) and fill the slot at match time.

Usage::

    from benchmark.dataset import load
    bundle = load("intents-for-eval")        # or "massive"
    bundle.intents      # {intent_id: {"train", "test_match", "entities", "domain"}}
    bundle.entities     # {slot_name: [example_value, ...]}
    bundle.domains      # {domain: [intent_id, ...]}
    bundle.no_match     # [utterance, ...] that should match nothing
    bundle.splits       # {split_name: [(utterance, expected_intent_or_None), ...]}
"""
from collections import defaultdict
from typing import NamedTuple

from datasets import load_dataset

#: short name -> Hugging Face repo id
DATASETS = {
    "intents-for-eval": "OpenVoiceOS/intents-for-eval",
    "massive": "OpenVoiceOS/massive-templates",
}

#: test splits whose utterances should match nothing
_NOMATCH_SPLITS = ("far_ood",)


class Bundle(NamedTuple):
    """A loaded benchmark dataset."""
    name: str
    repo: str
    lang: str
    intents: dict
    entities: dict
    domains: dict
    no_match: list
    splits: dict


def load(name: str = "intents-for-eval", lang: str = "en-US") -> Bundle:
    """Load a benchmark dataset from the Hugging Face Hub.

    Args:
        name: One of :data:`DATASETS` (``intents-for-eval`` or ``massive``).
        lang: BCP-47 language tag — the dataset's config prefix.

    Returns:
        A :class:`Bundle` with the templates grouped by intent and the test
        utterances split into matches / no-matches.
    """
    repo = DATASETS[name]
    templates = load_dataset(repo, f"{lang}-templates")["train"]
    test = load_dataset(repo, f"{lang}-test")["test"]

    intents: dict = {}
    domains: dict = defaultdict(list)
    entities: dict = defaultdict(list)
    for row in templates:
        iid = row["intent_id"]
        if iid not in intents:
            intents[iid] = {"train": [], "test_match": [],
                            "entities": [], "domain": row["domain"]}
            domains[row["domain"]].append(iid)
        if row["template"] not in intents[iid]["train"]:
            intents[iid]["train"].append(row["template"])
        for slot in row["slots"] or []:
            slot_name = slot["name"]
            if slot_name not in intents[iid]["entities"]:
                intents[iid]["entities"].append(slot_name)
            for example in slot["examples"] or []:
                if example not in entities[slot_name]:
                    entities[slot_name].append(example)

    no_match: list = []
    splits: dict = defaultdict(list)
    has_split = "split" in test.column_names
    for row in test:
        utt = row["utterance"]
        expected = row["expected_intent"] or None
        split = row["split"] if has_split else "test"
        if split in _NOMATCH_SPLITS or expected is None:
            no_match.append(utt)
            splits[split].append((utt, None))
        else:
            if expected in intents:
                intents[expected]["test_match"].append(utt)
            splits[split].append((utt, expected))

    return Bundle(name, repo, lang, intents, dict(entities), dict(domains),
                  no_match, dict(splits))
