"""`kab` and `kab-DZ` must land in the same locale bucket.

Kabyle ships under both spellings across the corpus (the skills register
under bare `kab`; `kab-DZ` is a duplicated code some `.intent` trees still
carry for historical reasons) -- they denote one language and must never
train as two disjoint labels.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from build_dataset import norm_lang  # noqa: E402


def test_kab_and_kab_dz_merge_to_one_bucket():
    assert norm_lang("kab") == "kab"
    assert norm_lang("kab-DZ") == "kab"
    assert norm_lang("kab-dz") == "kab"


def test_other_bare_and_hyphenated_codes_are_unaffected():
    assert norm_lang("en-us") == "en-US"
    assert norm_lang("an") == "an-ES"
    assert norm_lang("ca") == "ca-ES"
    assert norm_lang("de-de") == "de-DE"
