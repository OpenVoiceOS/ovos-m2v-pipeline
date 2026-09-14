"""`publish_corpus.py` uploads one staged corpus directory holding both splits.

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


def _write(root: Path, lang: str, filename: str, count: int) -> None:
    d = root / lang
    d.mkdir(exist_ok=True)
    lines = [json.dumps({"lang": lang, "label": "a:b", "utterance": str(i)})
             for i in range(count)]
    (d / filename).write_text("\n".join(lines) + ("\n" if lines else ""),
                              encoding="utf-8")


def _stage(tmp_path: Path, train_rows=None, test_rows=None) -> Path:
    """One tree, both splits, the way the dataset ships."""
    train_rows = train_rows if train_rows is not None else {"en-US": 3, "kab": 2}
    test_rows = test_rows if test_rows is not None else {"en-US": 2}
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "README.md").write_text("card", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"train_rows": 0}), encoding="utf-8")
    for lang, count in train_rows.items():
        _write(root, lang, publish_corpus.TEMPLATES_FILE, count)
    for lang, count in test_rows.items():
        _write(root, lang, publish_corpus.EVAL_FILE, count)
    return root


def _argv(root, *extra):
    return ["--corpus-dir", str(root), "--repo", "OpenVoiceOS/fake-corpus", *extra]


def test_locale_set_comes_from_the_directory(tmp_path):
    root = _stage(tmp_path, train_rows={"en-US": 1, "pt-PT": 2, "kab": 3})
    found = publish_corpus.inspect(root, publish_corpus.TEMPLATES_FILE)
    assert found["per_locale"] == {"en-US": 1, "kab": 3, "pt-PT": 2}
    assert found["rows"] == 6


def test_the_two_splits_are_read_from_the_same_tree(tmp_path):
    root = _stage(tmp_path)
    templates = publish_corpus.inspect(root, publish_corpus.TEMPLATES_FILE)
    gold = publish_corpus.inspect(root, publish_corpus.EVAL_FILE)
    assert templates["root"] == gold["root"] == root
    assert set(templates["per_locale"]) == {"en-US", "kab"}
    assert set(gold["per_locale"]) == {"en-US"}


@patch("huggingface_hub.HfApi")
def test_dry_run_uploads_nothing(mock_api_cls, tmp_path, capsys):
    root = _stage(tmp_path)
    assert publish_corpus.main(_argv(root, "--dry-run")) == 0
    mock_api_cls.assert_not_called()
    assert "nothing uploaded" in capsys.readouterr().out


@patch("huggingface_hub.HfApi")
def test_one_repo_takes_one_upload(mock_api_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    root = _stage(tmp_path)

    assert publish_corpus.main(_argv(root)) == 0

    assert mock_api.upload_folder.call_count == 1
    call = mock_api.upload_folder.call_args
    assert call.kwargs["repo_id"] == "OpenVoiceOS/fake-corpus"
    assert call.kwargs["repo_type"] == "dataset"
    assert call.kwargs["folder_path"] == str(root)


@patch("huggingface_hub.HfApi")
def test_the_tag_is_put_on_the_published_commit(mock_api_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    root = _stage(tmp_path)

    assert publish_corpus.main(_argv(root, "--tag", "v6")) == 0

    mock_api.create_tag.assert_called_once()
    assert mock_api.create_tag.call_args.kwargs["tag"] == "v6"
    assert mock_api.create_tag.call_args.kwargs["exist_ok"] is False


@patch("huggingface_hub.HfApi")
def test_no_tag_is_created_when_none_is_asked_for(mock_api_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    root = _stage(tmp_path)

    assert publish_corpus.main(_argv(root)) == 0

    mock_api.create_tag.assert_not_called()


@patch("huggingface_hub.HfApi")
def test_missing_token_refuses_to_upload(mock_api_cls, tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    root = _stage(tmp_path)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        publish_corpus.main(_argv(root))
    mock_api_cls.assert_not_called()


@patch("huggingface_hub.HfApi")
def test_a_locale_without_gold_is_reported_not_invented(mock_api_cls, tmp_path, capsys):
    root = _stage(tmp_path)
    publish_corpus.main(_argv(root, "--dry-run"))
    out = capsys.readouterr().out
    assert "1 locales have training rows and no gold (2 rows): kab" in out
    assert not (root / "kab" / publish_corpus.EVAL_FILE).exists()


@patch("huggingface_hub.HfApi")
def test_gold_without_training_rows_is_fatal(mock_api_cls, tmp_path):
    root = _stage(tmp_path, test_rows={"en-US": 2, "ja-JP": 1})
    with pytest.raises(SystemExit, match="ja-JP"):
        publish_corpus.main(_argv(root, "--dry-run"))


@patch("huggingface_hub.HfApi")
def test_an_empty_locale_file_is_fatal(mock_api_cls, tmp_path):
    root = _stage(tmp_path, test_rows={"en-US": 0})
    with pytest.raises(SystemExit, match="empty"):
        publish_corpus.main(_argv(root, "--dry-run"))


@patch("huggingface_hub.HfApi")
def test_a_missing_card_is_fatal(mock_api_cls, tmp_path):
    root = _stage(tmp_path)
    (root / "README.md").unlink()
    with pytest.raises(SystemExit, match="README.md"):
        publish_corpus.main(_argv(root, "--dry-run"))


@patch("huggingface_hub.HfApi")
def test_a_row_count_that_disagrees_with_the_build_is_fatal(mock_api_cls, tmp_path):
    root = _stage(tmp_path)
    with pytest.raises(SystemExit, match="expected 99"):
        publish_corpus.main(_argv(root, "--dry-run", "--expect-train-rows", "99"))


@patch("huggingface_hub.HfApi")
def test_the_expected_row_count_passes_when_it_agrees(mock_api_cls, tmp_path):
    root = _stage(tmp_path)
    assert publish_corpus.main(
        _argv(root, "--dry-run", "--expect-train-rows", "5",
              "--expect-test-rows", "2")) == 0


@patch("huggingface_hub.HfApi")
def test_a_failed_check_uploads_nothing(mock_api_cls, tmp_path, monkeypatch):
    """The whole tree is checked before the single upload, so a bad split
    cannot leave the repo holding the other one."""
    monkeypatch.setenv("HF_TOKEN", "t")
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    root = _stage(tmp_path, test_rows={"en-US": 2, "ja-JP": 1})
    with pytest.raises(SystemExit, match="ja-JP"):
        publish_corpus.main(_argv(root))
    mock_api.upload_folder.assert_not_called()
    mock_api.create_tag.assert_not_called()
