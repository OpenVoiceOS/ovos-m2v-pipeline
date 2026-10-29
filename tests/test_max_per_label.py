"""`train.py --max-per-label` caps skewed labels, stratified by language."""
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import train as train_mod  # noqa: E402


def _skewed_dataset(tmp_path: Path) -> Path:
    """One label with 100 rows (70 en-US / 30 pt-PT), one label with 5 rows."""
    rows = []
    for i in range(70):
        rows.append({"lang": "en-US", "label": "weather:weather",
                     "utterance": f"weather en {i}", "source": "corpus",
                     "skill_id": "skill-weather", "family": "weather"})
    for i in range(30):
        rows.append({"lang": "pt-PT", "label": "weather:weather",
                     "utterance": f"tempo pt {i}", "source": "corpus",
                     "skill_id": "skill-weather", "family": "weather"})
    for i in range(5):
        rows.append({"lang": "en-US", "label": "alarm:set",
                     "utterance": f"set alarm {i}", "source": "corpus",
                     "skill_id": "skill-alarm", "family": "alarm"})
    train_df = pd.DataFrame(rows)
    test_df = pd.DataFrame([
        {"lang": "en-US", "label": "weather:weather", "utterance": "weather test",
         "source": "corpus", "skill_id": "skill-weather", "family": "weather"},
        {"lang": "en-US", "label": "alarm:set", "utterance": "alarm test",
         "source": "corpus", "skill_id": "skill-alarm", "family": "alarm"},
    ])
    out = tmp_path / "dataset"
    out.mkdir()
    train_df.to_parquet(out / "train.parquet")
    test_df.to_parquet(out / "test.parquet")
    return out


def test_cap_skewed_label_stratified_by_lang(tmp_path):
    dataset = _skewed_dataset(tmp_path)
    train, test = train_mod.load(dataset, None, None)

    capped, report = train_mod.cap_per_label(train, max_per_label=10, seed=0)

    weather = capped[capped["label"] == "weather:weather"]
    assert len(weather) == 10
    assert (weather["lang"] == "en-US").sum() == 7
    assert (weather["lang"] == "pt-PT").sum() == 3
    assert report["seed"] == 0
    assert report["max_per_label"] == 10
    assert report["before"]["weather:weather"] == 100
    assert report["after"]["weather:weather"] == 10


def test_cap_is_deterministic_for_same_seed_and_varies_by_seed(tmp_path):
    dataset = _skewed_dataset(tmp_path)
    train, _ = train_mod.load(dataset, None, None)

    capped_a, _ = train_mod.cap_per_label(train, max_per_label=10, seed=0)
    capped_b, _ = train_mod.cap_per_label(train, max_per_label=10, seed=0)
    capped_c, _ = train_mod.cap_per_label(train, max_per_label=10, seed=1)

    weather_a = capped_a[capped_a["label"] == "weather:weather"].sort_values("utterance")
    weather_b = capped_b[capped_b["label"] == "weather:weather"].sort_values("utterance")
    weather_c = capped_c[capped_c["label"] == "weather:weather"].sort_values("utterance")

    assert weather_a["utterance"].tolist() == weather_b["utterance"].tolist()
    assert weather_a["utterance"].tolist() != weather_c["utterance"].tolist()


def test_label_below_cap_is_untouched(tmp_path):
    dataset = _skewed_dataset(tmp_path)
    train, _ = train_mod.load(dataset, None, None)

    capped, report = train_mod.cap_per_label(train, max_per_label=10, seed=0)

    alarm = capped[capped["label"] == "alarm:set"]
    assert len(alarm) == 5
    assert report["before"]["alarm:set"] == 5
    assert report["after"]["alarm:set"] == 5


def test_test_split_is_never_touched(tmp_path):
    dataset = _skewed_dataset(tmp_path)
    train, test = train_mod.load(dataset, None, None)
    original_test = test.copy()

    train_mod.cap_per_label(train, max_per_label=10, seed=0)

    pd.testing.assert_frame_equal(test.reset_index(drop=True),
                                  original_test.reset_index(drop=True))


def test_cap_selection_is_invariant_to_row_order(tmp_path):
    dataset = _skewed_dataset(tmp_path)
    train, _ = train_mod.load(dataset, None, None)
    shuffled = train.sample(frac=1, random_state=7).reset_index(drop=True)

    capped_a, _ = train_mod.cap_per_label(train, max_per_label=10, seed=0)
    capped_b, _ = train_mod.cap_per_label(shuffled, max_per_label=10, seed=0)

    weather_a = set(capped_a[capped_a["label"] == "weather:weather"]["utterance"])
    weather_b = set(capped_b[capped_b["label"] == "weather:weather"]["utterance"])
    assert weather_a == weather_b


def test_no_cap_argument_leaves_rows_untouched():
    args = train_mod.build_arg_parser().parse_args(["--dataset", "unused"])
    assert args.max_per_label is None
    assert args.seed == 0
