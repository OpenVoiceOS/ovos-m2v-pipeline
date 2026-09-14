"""`publish_corpus.py` uploads the two staged corpus directories.

`HfApi` is mocked throughout: these tests must never touch the network.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("huggingface_hub")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import publish_corpus  # noqa: E402


def _stage(tmp_path: Path, name: str, filename: str, rows: dict) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "README.md").write_text("card", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"train_rows": 0}), encoding="utf-8")
    for lang, count in rows.items():
        d = root / lang
        d.mkdir()
        lines = [json.dumps({"lang": lang, "label": "a:b", "utterance": str(i)})
                 for i in range(count)]
        (d / filename).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return root


def _pair(tmp_path, train_rows=None, test_rows=None):
    train_rows = train_rows if train_rows is not None else {"en-US": 3, "kab": 2}
    test_rows = test_rows if test_rows is not None else {"en-US": 2}
    tpl = _stage(tmp_path, "tpl", publish_corpus.TEMPLATES_FILE, train_rows)
    ev = _stage(tmp_path, "ev", publish_corpus.EVAL_FILE, test_rows)
    return tpl, ev


def _argv(tpl, ev, *extra):
    return ["--templates-dir", str(tpl), "--eval-dir", str(ev),
            "--templates-repo", "OpenVoiceOS/fake-templates",
            "--eval-repo", "OpenVoiceOS/fake-eval", *extra]


def test_locale_set_comes_from_the_directory(tmp_path):
    tpl, _ = _pair(tmp_path, train_rows={"en-US": 1, "pt-PT": 2, "kab": 3})
    found = publish_corpus.inspect(tpl, publish_corpus.TEMPLATES_FILE)
    assert found["per_locale"] == {"en-US": 1, "kab": 3, "pt-PT": 2}
    assert found["rows"] == 6


@patch("huggingface_hub.HfApi")
def test_dry_run_uploads_nothing(mock_api_cls, tmp_path, capsys):
    tpl, ev = _pair(tmp_path)
    assert publish_corpus.main(_argv(tpl, ev, "--dry-run")) == 0
    mock_api_cls.assert_not_called()
    assert "nothing uploaded" in capsys.readouterr().out


@patch("huggingface_hub.HfApi")
def test_both_repos_are_uploaded(mock_api_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    tpl, ev = _pair(tmp_path)

    assert publish_corpus.main(_argv(tpl, ev)) == 0

    repos = [c.kwargs["repo_id"] for c in mock_api.upload_folder.call_args_list]
    assert repos == ["OpenVoiceOS/fake-templates", "OpenVoiceOS/fake-eval"]
    assert all(c.kwargs["repo_type"] == "dataset"
               for c in mock_api.upload_folder.call_args_list)


@patch("huggingface_hub.HfApi")
def test_missing_token_refuses_to_upload(mock_api_cls, tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    tpl, ev = _pair(tmp_path)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        publish_corpus.main(_argv(tpl, ev))
    mock_api_cls.assert_not_called()


@patch("huggingface_hub.HfApi")
def test_a_locale_without_gold_is_reported_not_invented(mock_api_cls, tmp_path, capsys):
    tpl, ev = _pair(tmp_path)
    publish_corpus.main(_argv(tpl, ev, "--dry-run"))
    out = capsys.readouterr().out
    assert "1 locales have training rows and no gold (2 rows): kab" in out
    assert not (ev / "kab").exists()


@patch("huggingface_hub.HfApi")
def test_gold_without_training_rows_is_fatal(mock_api_cls, tmp_path):
    tpl, ev = _pair(tmp_path, test_rows={"en-US": 2, "ja-JP": 1})
    with pytest.raises(SystemExit, match="ja-JP"):
        publish_corpus.main(_argv(tpl, ev, "--dry-run"))


@patch("huggingface_hub.HfApi")
def test_an_empty_locale_file_is_fatal(mock_api_cls, tmp_path):
    tpl, ev = _pair(tmp_path, test_rows={"en-US": 0})
    with pytest.raises(SystemExit, match="empty"):
        publish_corpus.main(_argv(tpl, ev, "--dry-run"))


@patch("huggingface_hub.HfApi")
def test_a_missing_card_is_fatal(mock_api_cls, tmp_path):
    tpl, ev = _pair(tmp_path)
    (tpl / "README.md").unlink()
    with pytest.raises(SystemExit, match="README.md"):
        publish_corpus.main(_argv(tpl, ev, "--dry-run"))


@patch("huggingface_hub.HfApi")
def test_a_row_count_that_disagrees_with_the_build_is_fatal(mock_api_cls, tmp_path):
    tpl, ev = _pair(tmp_path)
    with pytest.raises(SystemExit, match="expected 99"):
        publish_corpus.main(_argv(tpl, ev, "--dry-run", "--expect-train-rows", "99"))


@patch("huggingface_hub.HfApi")
def test_the_expected_row_count_passes_when_it_agrees(mock_api_cls, tmp_path):
    tpl, ev = _pair(tmp_path)
    assert publish_corpus.main(
        _argv(tpl, ev, "--dry-run", "--expect-train-rows", "5",
              "--expect-test-rows", "2")) == 0
