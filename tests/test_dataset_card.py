"""The dataset card must not overstate what the corpus covers.

A reader deciding whether to use the corpus for their language sees the card
before anything else, so the census belongs in it rather than in a companion
file, and a language tag attested by one skill's templates must not be
counted as coverage.
"""
import sys
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[1] / "train"
sys.path.insert(0, str(BUILDER))

from build_dataset import (DATASET_FILES, MIN_SKILLS_FOR_COVERAGE,  # noqa: E402
                          write_dataset_card)

REPORT = {
    "rows_final": 1000,
    "labels": 10,
    "rows_per_lang": {"en": 900, "pt": 100},
    "skills_per_lang": {"en": 40, "pt": 12},
    "labels_per_lang": {"en": 10, "pt": 9},
    "variants_per_lang": {"en": ["en-US"], "pt": ["pt-BR", "pt-PT"]},
    "languages_excluded_from_publication": {
        "thin_languages": {"languages": [], "rows": 0}},
    "dropped_unresolved_labels": 25,
    "dropped_rare_labels": {"labels": 3, "examples": []},
    "labels_without_test_rows": {"labels": 2, "names": ["a:b", "c:d"]},
}


def card(tmp_path) -> str:
    write_dataset_card(tmp_path, REPORT)
    return (tmp_path / "README.md").read_text(encoding="utf-8")


def test_the_card_ships_with_the_dataset():
    assert "README.md" in DATASET_FILES


def test_the_card_says_it_is_language_level(tmp_path):
    """`pt` must not be read as European Portuguese, and the tag itself
    cannot say so."""
    text = card(tmp_path)
    assert "language-level dataset" in text
    assert "Read `es` as Spanish, not as any one region's Spanish." in text


def test_the_card_names_the_variants_behind_each_language(tmp_path):
    text = card(tmp_path)
    assert "| `pt` | `pt-BR`, `pt-PT` |" in text


def test_the_card_says_the_skills_keep_their_dialects(tmp_path):
    """Grouping the corpus is not the project dropping dialect support, and
    those are very different claims."""
    text = card(tmp_path)
    assert "ship their dialect locales separately" in text
    assert "lang_variant" in text


def test_the_card_gives_label_coverage_beside_rows(tmp_path):
    text = card(tmp_path)
    assert "| `pt` | 100 | 9 | 12 |" in text


def test_every_published_language_is_listed_with_its_row_count(tmp_path):
    text = card(tmp_path)
    for lang, rows in REPORT["rows_per_lang"].items():
        assert f"| `{lang}` | {rows:,} |" in text


def test_the_card_states_the_skew(tmp_path):
    assert "90% of the corpus" in card(tmp_path)


def test_the_card_states_which_labels_no_number_can_score(tmp_path):
    text = card(tmp_path)
    assert "2 of 10 labels have no rows in the test split" in text


def test_the_card_states_what_was_dropped(tmp_path):
    text = card(tmp_path)
    assert "25 rows were dropped" in text
    assert "3 further labels" in text


