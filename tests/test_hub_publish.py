"""`build_dataset.py --push-to` / `train.py --push-to` upload to the Hub.

`HfApi` is mocked throughout: these tests must never touch the network.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pandas")
pytest.importorskip("huggingface_hub")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_dataset  # noqa: E402
import train as train_mod  # noqa: E402


def _dataset_dir(tmp_path: Path) -> Path:
    out = tmp_path / "dataset"
    out.mkdir()
    for name in build_dataset.DATASET_FILES:
        (out / name).write_text("x", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(
        {"outputs": {"train.parquet": "sha-train", "test.parquet": "sha-test",
                     "labels.json": "sha-labels", "manifest.json": "sha-self"}}),
        encoding="utf-8")
    return out


@patch("huggingface_hub.HfApi")
def test_push_dataset_uploads_every_file(mock_api_cls, tmp_path):
    out = _dataset_dir(tmp_path)
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    build_dataset.push_dataset(out, "OpenVoiceOS/fake-dataset", dry_run=False)

    mock_api_cls.assert_called_once()
    mock_api.create_repo.assert_called_once_with(
        "OpenVoiceOS/fake-dataset", repo_type="dataset", exist_ok=True)
    uploaded = {c.kwargs["path_in_repo"] for c in mock_api.upload_file.call_args_list}
    assert uploaded == set(build_dataset.DATASET_FILES)
    for c in mock_api.upload_file.call_args_list:
        assert c.kwargs["repo_id"] == "OpenVoiceOS/fake-dataset"
        assert c.kwargs["repo_type"] == "dataset"


@patch("huggingface_hub.HfApi")
def test_push_dataset_dry_run_uploads_nothing(mock_api_cls, tmp_path, capsys):
    out = _dataset_dir(tmp_path)

    build_dataset.push_dataset(out, "OpenVoiceOS/fake-dataset", dry_run=True)

    mock_api_cls.assert_not_called()
    out_text = capsys.readouterr().out
    assert "would upload" in out_text
    for name in build_dataset.DATASET_FILES:
        assert name in out_text


@patch("huggingface_hub.HfApi")
def test_push_dataset_missing_file_refuses_to_push(mock_api_cls, tmp_path):
    out = tmp_path / "dataset"
    out.mkdir()
    (out / "train.parquet").write_text("x", encoding="utf-8")  # rest missing

    with pytest.raises(SystemExit):
        build_dataset.push_dataset(out, "OpenVoiceOS/fake-dataset", dry_run=False)
    mock_api_cls.assert_not_called()


def _model_out_dir(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps(
        {"outputs": {"train.parquet": "sha-train", "test.parquet": "sha-test"}}),
        encoding="utf-8")
    out = tmp_path / "model_out"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")
    (out / "pipeline.skops").write_bytes(b"fake-skops")
    (out / "tokenizer.json").write_text("{}", encoding="utf-8")
    onnx = out / "onnx"
    onnx.mkdir()
    (onnx / "model.onnx").write_bytes(b"fake-onnx")
    return out, dataset


@patch("huggingface_hub.HfApi")
def test_push_model_uploads_the_whole_tree(mock_api_cls, tmp_path):
    out, dataset = _model_out_dir(tmp_path)
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    train_mod.push_model(out, dataset, "OpenVoiceOS/fake-model", dry_run=False)

    mock_api_cls.assert_called_once()
    mock_api.create_repo.assert_called_once_with(
        "OpenVoiceOS/fake-model", repo_type="model", exist_ok=True)
    mock_api.upload_folder.assert_called_once()
    call = mock_api.upload_folder.call_args
    assert call.kwargs["folder_path"] == str(out)
    assert call.kwargs["repo_id"] == "OpenVoiceOS/fake-model"
    assert call.kwargs["repo_type"] == "model"


@patch("huggingface_hub.HfApi")
def test_push_model_dry_run_uploads_nothing(mock_api_cls, tmp_path, capsys):
    out, dataset = _model_out_dir(tmp_path)

    train_mod.push_model(out, dataset, "OpenVoiceOS/fake-model", dry_run=True)

    mock_api_cls.assert_not_called()
    out_text = capsys.readouterr().out
    assert "would upload" in out_text
    assert "config.json" in out_text
    assert "onnx/model.onnx" in out_text


def test_training_manifest_carries_dataset_shas_and_model2vec_version(tmp_path):
    out, dataset = _model_out_dir(tmp_path)

    train_mod.push_model(out, dataset, "OpenVoiceOS/fake-model", dry_run=True)

    manifest = json.loads((out / "training_manifest.json").read_text())
    assert manifest["dataset_shas"] == {"train.parquet": "sha-train",
                                        "test.parquet": "sha-test"}
    import model2vec
    assert manifest["model2vec_version"] == model2vec.__version__
