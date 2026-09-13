"""Calibration script for prototype-mode confidence thresholds.

Loads the default multilingual model, registers a handful of intents as
prototype templates, and reports the cosine-similarity distribution for
matching vs. non-matching utterances. Used to derive the `conf_high`,
`conf_medium`, and `conf_low` defaults for `Model2VecPrototypePipeline`.

Run with: python3 scripts/calibrate_prototype_thresholds.py
"""
import numpy as np
from model2vec import StaticModel

DEFAULT_MULTILINGUAL = "OpenVoiceOS/ovos-m2v-intents-multilingual"

TEMPLATES = {
    "hello": [
        "hello", "hi there", "good morning", "hey", "greetings",
    ],
    "weather": [
        "what's the weather like", "is it going to rain today",
        "tell me the forecast", "how hot is it outside",
    ],
    "timer": [
        "set a timer for five minutes", "start a timer",
        "cancel the timer", "how much time is left on the timer",
    ],
    "play_music": [
        "play some music", "play a song by daft punk",
        "put on some jazz", "resume the music",
    ],
}

MATCHING = {
    "hello": ["hi", "good afternoon", "hey there"],
    "weather": ["will it rain tomorrow", "what's the temperature outside"],
    "timer": ["set a timer for ten minutes", "stop the timer"],
    "play_music": ["play the beatles", "turn on some music"],
}

NON_MATCHING = [
    "what time is it",
    "turn off the lights",
    "tell me a joke",
    "what's on my calendar today",
    "remind me to buy milk",
    "how far away is the moon",
    "translate hello to spanish",
    "open the garage door",
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    model = StaticModel.from_pretrained(DEFAULT_MULTILINGUAL)

    proto_embs = {}
    for label, sents in TEMPLATES.items():
        embs = model.encode(sents)
        proto_embs[label] = embs

    def best_score(utt_emb):
        best_label, best_score = None, -1.0
        for label, embs in proto_embs.items():
            for e in embs:
                s = cosine(utt_emb, e)
                if s > best_score:
                    best_score, best_label = s, label
        return best_label, best_score

    match_scores = []
    for gold_label, utts in MATCHING.items():
        for u in utts:
            emb = model.encode([u])[0]
            pred_label, score = best_score(emb)
            match_scores.append(score)
            print(f"MATCH   {gold_label!r:15} -> {pred_label!r:15} score={score:.4f} utt={u!r}")

    nonmatch_scores = []
    for u in NON_MATCHING:
        emb = model.encode([u])[0]
        pred_label, score = best_score(emb)
        nonmatch_scores.append(score)
        print(f"NOMATCH {'':15} -> {pred_label!r:15} score={score:.4f} utt={u!r}")

    match_scores = np.array(match_scores)
    nonmatch_scores = np.array(nonmatch_scores)

    print()
    print(f"matching:     min={match_scores.min():.4f} mean={match_scores.mean():.4f} "
          f"median={np.median(match_scores):.4f} max={match_scores.max():.4f}")
    print(f"non-matching: min={nonmatch_scores.min():.4f} mean={nonmatch_scores.mean():.4f} "
          f"median={np.median(nonmatch_scores):.4f} max={nonmatch_scores.max():.4f}")


if __name__ == "__main__":
    main()
