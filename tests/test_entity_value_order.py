"""Entity values are stored in one order, whatever the interpreter's hash seed.

The values of an entity fill the ``{slot}`` of template intents, and the order
of the filled samples decides which prototypes the store keeps as anchors. An
order that follows ``PYTHONHASHSEED`` makes the score of one utterance differ
from process to process.
"""
import json
import os
import subprocess
import sys
import textwrap
import unittest

VALUES = ["friday", "monday", "15th of May", "1st of April 2023", "weekend",
          "12/25/2024", "tomorrow", "next week", "march", "2022"]

SCRIPT = textwrap.dedent("""
    import json, sys
    from unittest import mock
    from ovos_bus_client.message import Message
    from ovos_spec_tools import SpecMessage
    from ovos_utils.fakebus import FakeBus
    from ovos_m2v_pipeline import Model2VecPrototypePipeline

    with mock.patch.object(Model2VecPrototypePipeline, "_ensure_model",
                           lambda self, **kw: None):
        pipe = Model2VecPrototypePipeline(FakeBus(), {})
    pipe._handle_intent4_register_entity(Message(
        SpecMessage.ENTITY_REGISTER.value,
        {"skill_id": "a.skill", "entity_name": "date", "lang": "en-US",
         "samples": json.loads(sys.argv[1])}))
    print(json.dumps(pipe.entities["a.skill"]["en-US"]["date"]))
""")


def _stored_under(seed: int) -> list:
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run([sys.executable, "-c", SCRIPT, json.dumps(VALUES)],
                         env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


class EntityValueOrderTest(unittest.TestCase):
    def test_stored_values_are_sorted_under_every_hash_seed(self):
        for seed in (0, 1, 2, 3):
            self.assertEqual(_stored_under(seed), sorted(VALUES), f"seed {seed}")
