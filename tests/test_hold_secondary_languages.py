"""While skills load, a secondary language's prototypes are held: they are
added once registrations have gone quiet, or when a query in that language
needs them. The primary language and ``eager_langs`` are never held."""
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from ovos_bus_client.message import Message

from ovos_m2v_pipeline import Model2VecIntentPipeline
from tests.test_intent4 import _make_prototype_pipeline

SKILL = "clock.skill"


def register(p, lang, name="time", samples=("what time is it",)):
    p._handle_register_padatious(Message(
        "padatious:register_intent",
        {"name": f"{SKILL}:{name}.intent", "samples": list(samples), "lang": lang},
        {"skill_id": SKILL}))


def booting(config=None):
    """A pipeline still loading skills (the test factory settles its boot)."""
    p = _make_prototype_pipeline(config)
    p._boot_settled.clear()
    p.model.encode.side_effect = (
        lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32))
    return p


class TestHoldSecondaryLanguages(unittest.TestCase):
    def test_primary_is_added_secondary_is_held(self):
        p = booting()
        register(p, "en-US")
        register(p, "fr-FR", samples=("quelle heure est-il",))
        self.assertEqual(len(p.prototype_store), 1)
        self.assertEqual(len(p._held_additions["fr-FR"]), 1)

    def test_settling_adds_what_was_held_in_order(self):
        p = booting()
        register(p, "fr-FR", samples=("quelle heure est-il",))
        register(p, "fr-FR", name="date", samples=("quel jour sommes-nous",))
        p._settle_boot()
        self.assertEqual(p._held_additions, {})
        self.assertEqual(len(p.prototype_store), 2)
        # once settled, a registration goes straight in
        register(p, "fr-FR", name="year", samples=("quelle année",))
        self.assertEqual(len(p.prototype_store), 3)

    def test_a_query_adds_its_language_first(self):
        p = booting()
        register(p, "fr-FR", samples=("quelle heure est-il",))
        list(p._match_prototype("quelle heure est-il", lang="fr-fr"))
        self.assertNotIn("fr-FR", p._held_additions)
        self.assertEqual(len(p.prototype_store), 1)
        self.assertFalse(p._boot_settled.is_set())

    def test_a_detach_drops_the_held_registration(self):
        p = booting()
        register(p, "fr-FR")
        p._handle_detach_intent(Message("detach_intent",
                                        {"intent_name": f"{SKILL}:time.intent"}))
        self.assertEqual(p._held_additions["fr-FR"], [])
        p._settle_boot()
        self.assertEqual(len(p.prototype_store), 0)

    def test_eager_languages_are_not_held(self):
        p = booting({"eager_langs": ["fr-fr"]})
        register(p, "fr-FR")
        self.assertNotIn("fr-FR", p._held_additions)
        self.assertEqual(len(p.prototype_store), 1)

    def test_quiet_registrations_settle_by_themselves(self):
        with patch.object(Model2VecIntentPipeline, "_SECONDARY_QUIET_S", 0.5):
            p = booting()
            threading.Thread(target=p._watch_boot, daemon=True).start()
            register(p, "fr-FR")
            deadline = time.monotonic() + 10
            while not p._boot_settled.is_set() and time.monotonic() < deadline:
                time.sleep(0.1)
        self.assertTrue(p._boot_settled.is_set())
        self.assertEqual(len(p.prototype_store), 1)


if __name__ == "__main__":
    unittest.main()


class TestHoldEdges(unittest.TestCase):
    def test_every_registration_path_counts_as_activity(self):
        p = booting()
        p._last_registration_at = None
        p._handle_register_adapt(Message("register_intent", {"name": f"{SKILL}:adapt"},
                                         {"skill_id": SKILL}))
        self.assertIsNotNone(p._last_registration_at)

    def test_a_held_language_skips_the_prebuilt_shortcut(self):
        p = booting()
        p._prebuilt_validated = True
        with patch.object(p, "_prototype_cache_key", return_value="k"), \
                patch.object(p, "_install_prebuilt_label", return_value=True) as prebuilt:
            p._prebuilt_cache_keys = {f"{SKILL}:time": "k"}
            register(p, "fr-FR")
            prebuilt.assert_not_called()
            self.assertEqual(len(p._held_additions["fr-FR"]), 1)
            register(p, "en-US")
            prebuilt.assert_called_once()

    def test_a_query_in_a_dialect_adds_the_held_language_it_scores_against(self):
        p = booting()
        register(p, "fr-FR", samples=("quelle heure est-il",))
        list(p._match_prototype("quelle heure est-il", lang="fr-ca"))
        self.assertNotIn("fr-FR", p._held_additions)
        self.assertEqual(len(p.prototype_store), 1)
