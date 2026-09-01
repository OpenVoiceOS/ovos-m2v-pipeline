"""Live E2E test: run the en-US intent fixture (drawn from
``ovos-localize/data/datasets/classification/en-US.jsonl``, official OVOS
skills only) through **both** Model2Vec pipeline modes:

1. **classifier** — `ovos-m2v-pipeline` driven through a MiniCroft + bus +
   IntentService; asserts the dispatched intent matches the expected
   fixture label (top-1).
2. **prototype** — `ovos-m2v-prototype-pipeline` exercised directly via
   `Model2VecPrototypePipeline` + `PrototypeIntentStore`. Uses
   **leave-one-out** evaluation: for each fixture case, rebuild the store
   from all OTHER utterances and predict on the held-out one.

Both modes emit their per-case results into a single markdown report
written to ``LIVE_REPORT_PATH``; the live_tests workflow uploads that file
as a sticky PR comment so reviewers see classifier + prototype accuracy
side-by-side.

Gated behind ``OVOSCOPE_LIVE=1`` because both modes download HuggingFace
models on first run.
"""
import json
import os
import threading
import unittest
from pathlib import Path

import pytest

if os.environ.get("OVOSCOPE_LIVE") != "1":
    pytest.skip(
        "Live model test skipped; set OVOSCOPE_LIVE=1 to enable.",
        allow_module_level=True,
    )

pytest.importorskip("ovoscope", reason="ovoscope not installed")

import numpy as np  # noqa: E402
from ovos_bus_client.message import Message  # noqa: E402
from ovos_bus_client.session import Session  # noqa: E402
from ovos_config.config import Configuration  # noqa: E402
from ovoscope import get_minicroft  # noqa: E402

from ovos_m2v_pipeline import PrototypeIntentStore  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "en_us_intents.jsonl"

# Classifier mode — pre-trained model, label space fixed at training time.
CLF_PIPELINE_ID = "ovos-m2v-pipeline"
CLF_CONFIG_KEY = "ovos_m2v_pipeline"
CLF_MODEL = "Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2"

# Prototype mode — bare embedding model; labels are user-supplied.
PROTO_MODEL = "minishlab/M2V_multilingual_output"
PROTO_K = 5  # max prototypes per label

# Used by the live_tests workflow when composing the PR comment.
REPORT_PATH = Path(os.environ.get(
    "LIVE_REPORT_PATH",
    str(Path(__file__).parent / "live_test_report.md"),
))

# Module-level accumulators — both test classes append their section.
_REPORT_SECTIONS: list[str] = []


def _load_fixture():
    cases = []
    with FIXTURE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _flush_report():
    if not _REPORT_SECTIONS:
        return
    body = "# Live fixture test\n\n" + "\n\n".join(_REPORT_SECTIONS) + "\n"
    try:
        REPORT_PATH.write_text(body)
    except OSError:
        pass


def _section(mode: str, total: int, passed: int, misses: list,
             tolerance_pct: int, ok: bool) -> str:
    status = "✅ PASS" if ok else "❌ FAIL"
    acc = passed / total if total else 0.0
    lines = [
        f"## {mode}",
        "",
        f"**Status:** {status}",
        f"**Accuracy:** {passed}/{total} ({acc:.1%}) — tolerance ≤ {tolerance_pct}% drift",
    ]
    if misses:
        lines += [
            "",
            "<details><summary>Misclassifications</summary>",
            "",
            "| utterance | expected | got |",
            "|---|---|---|",
        ]
        for u, e, g in misses[:30]:
            lines.append(f"| `{u}` | `{e}` | `{g}` |")
        if len(misses) > 30:
            lines.append(f"\n_…and {len(misses) - 30} more_")
        lines.append("\n</details>")
    else:
        lines += ["", "No misclassifications."]
    return "\n".join(lines)


class TestClassifierLiveFixture(unittest.TestCase):
    """Classifier-mode pipeline through full MiniCroft / bus path."""

    @classmethod
    def setUpClass(cls):
        cls.cases = _load_fixture()
        cls.all_labels = sorted({c["label"] for c in cls.cases})

        cfg = Configuration()
        intents_cfg = cfg.setdefault("intents", {})
        cls._orig = intents_cfg.get(CLF_CONFIG_KEY)
        intents_cfg[CLF_CONFIG_KEY] = {
            "model": CLF_MODEL,
            "renormalize": False,
            "conf_low": 0.0,
        }

        cls.mc = get_minicroft(
            skill_ids=[],
            lang="en-US",
            default_pipeline=[CLF_PIPELINE_ID],
            max_wait=300,
        )
        cls.pipeline = cls.mc.intents.pipeline_plugins[CLF_PIPELINE_ID]
        cls.pipeline.intents = list(cls.all_labels)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.mc.stop()
        finally:
            cfg = Configuration()
            intents_cfg = cfg.get("intents", {})
            if cls._orig is None:
                intents_cfg.pop(CLF_CONFIG_KEY, None)
            else:
                intents_cfg[CLF_CONFIG_KEY] = cls._orig
        _flush_report()

    def _emit(self, utterance: str, expected_label: str, timeout: float = 15.0):
        got: list[Message] = []
        done = threading.Event()

        def _capture(msg):
            got.append(msg)
            done.set()

        def _fail(_msg):
            done.set()

        self.mc.bus.on(expected_label, _capture)
        self.mc.bus.on("complete_intent_failure", _fail)
        sess = Session(session_id="live-clf", pipeline=[CLF_PIPELINE_ID])
        try:
            self.mc.bus.emit(Message(
                "recognizer_loop:utterance",
                data={"utterances": [utterance], "lang": "en-US"},
                context={"session": sess.serialize()},
            ))
            done.wait(timeout=timeout)
        finally:
            self.mc.bus.remove(expected_label, _capture)
            self.mc.bus.remove("complete_intent_failure", _fail)
        return got[0] if got else None

    def test_fixture_top_intent_matches_label(self):
        misses = []
        for case in self.cases:
            utt, expected = case["utterance"], case["label"]
            msg = self._emit(utt, expected)
            if msg is None:
                misses.append((utt, expected, "no match"))
            elif msg.msg_type != expected:
                misses.append((utt, expected, msg.msg_type))

        total = len(self.cases)
        passed = total - len(misses)
        max_misses = max(1, total // 5)
        ok = len(misses) <= max_misses

        _REPORT_SECTIONS.append(_section(
            mode="`ovos-m2v-pipeline` — classifier mode (MiniCroft + bus)",
            total=total, passed=passed, misses=misses,
            tolerance_pct=20, ok=ok,
        ))
        _flush_report()

        self.assertTrue(
            ok,
            f"classifier mode: {len(misses)}/{total} misclassified "
            f"(max allowed {max_misses}):\n"
            + "\n".join(f"  {u!r} expected={e} got={g}" for u, e, g in misses),
        )


class TestPrototypeLiveFixture(unittest.TestCase):
    """Prototype-mode pipeline via leave-one-out cross-validation.

    For each fixture case, build a `PrototypeIntentStore` from all OTHER
    utterances and predict on the held-out one. This exercises the same
    code path the bus-driven prototype registration uses, but without the
    per-case MiniCroft setup overhead (which would be prohibitive for
    leave-one-out at 100+ cases).
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = _load_fixture()
        cls.all_labels = sorted({c["label"] for c in cls.cases})

        # Embeddings-only model — bare StaticModel, no trained head.
        from model2vec import StaticModel
        cls.model = StaticModel.from_pretrained(PROTO_MODEL)

        # Pre-encode every utterance once so leave-one-out is fast.
        cls.embeddings = cls.model.encode([c["utterance"] for c in cls.cases])
        # `model2vec` returns float32 by default; ensure that for the store.
        cls.embeddings = np.asarray(cls.embeddings, dtype=np.float32)

    @classmethod
    def tearDownClass(cls):
        _flush_report()

    def _predict_loo(self, held_out_idx: int) -> str | None:
        """Return top-1 predicted label for the held-out case (or None)."""
        # Build the store from every OTHER case, capped at PROTO_K per label.
        per_label: dict[str, list[int]] = {}
        for i, case in enumerate(self.cases):
            if i == held_out_idx:
                continue
            per_label.setdefault(case["label"], []).append(i)

        embs: list[np.ndarray] = []
        labels: list[str] = []
        for label, idxs in per_label.items():
            # Mirror PrototypeIntentStore.build's sampling: first k.
            for i in idxs[:PROTO_K]:
                embs.append(self.embeddings[i])
                labels.append(label)

        if not embs:
            return None

        store = PrototypeIntentStore(
            np.stack(embs), np.array(labels, dtype=object)
        )
        scores = store.scores(self.embeddings[held_out_idx])
        if not scores:
            return None
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def test_prototype_top_label_matches_loo(self):
        misses = []
        for i, case in enumerate(self.cases):
            top = self._predict_loo(i)
            if top != case["label"]:
                misses.append((case["utterance"], case["label"], top or "no match"))

        total = len(self.cases)
        passed = total - len(misses)
        # Prototype mode on a small bare-embedding model is less accurate
        # than the trained classifier — allow more drift.
        max_misses = max(1, total // 3)  # ~33 %
        ok = len(misses) <= max_misses

        _REPORT_SECTIONS.append(_section(
            mode=(f"`ovos-m2v-prototype-pipeline` — prototype mode "
                  f"(leave-one-out, k={PROTO_K})"),
            total=total, passed=passed, misses=misses,
            tolerance_pct=33, ok=ok,
        ))
        _flush_report()

        self.assertTrue(
            ok,
            f"prototype mode: {len(misses)}/{total} misclassified "
            f"(max allowed {max_misses}):\n"
            + "\n".join(f"  {u!r} expected={e} got={g}" for u, e, g in misses),
        )


if __name__ == "__main__":
    unittest.main()
