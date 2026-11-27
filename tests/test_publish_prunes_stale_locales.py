"""A publish must leave the repository holding this build and nothing else.

`upload_folder` creates and updates; on its own it deletes nothing. A locale
directory written by an earlier publish therefore outlives every build that
does not write it, and a consumer that walks the repository reads rows the
build never produced.

That is not hypothetical. `arb` and `es-419` reached the `v6` tag of
`OpenVoiceOS/ovos-intents-v5-templates` carrying the v5 row schema
(`intent_id`, `lang`, `template`, with no `label` and no `utterance`), while
the manifest of that same tag recorded 52 training languages against 54
locale directories in the repository. The adapter that read every directory
with one code path raised `KeyError` on the two.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import publish_corpus  # noqa: E402


def _stage(tmp_path: Path) -> Path:
    root = tmp_path / "staged"
    (root / "en-US").mkdir(parents=True)
    rows = [{"label": "a:one", "lang": "en-US", "source": "s",
             "template": "t", "utterance": "u"}]
    for name in ("train_templates.jsonl", "test.jsonl"):
        with (root / "en-US" / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    (root / "manifest.json").write_text(json.dumps({"train_rows": 1, "test_rows": 1}),
                                        encoding="utf-8")
    (root / "README.md").write_text("# card\n", encoding="utf-8")
    return root


def _argv(root: Path, *extra):
    return ["--corpus-dir", str(root), "--repo", "OpenVoiceOS/fake-corpus", *extra]


@patch("huggingface_hub.HfApi")
def test_the_upload_deletes_a_locale_payload_this_build_did_not_stage(
        mock_api_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    assert publish_corpus.main(_argv(_stage(tmp_path))) == 0

    patterns = mock_api.upload_folder.call_args.kwargs["delete_patterns"]
    assert patterns, "the upload deletes nothing, so a stale locale survives it"
    assert any(_matches(pattern, "arb/train_templates.jsonl")
               for pattern in patterns)
    assert any(_matches(pattern, "es-419/train_templates.jsonl")
               for pattern in patterns)


@patch("huggingface_hub.HfApi")
def test_the_repository_root_is_never_deleted(mock_api_cls, tmp_path, monkeypatch):
    """The card, the manifest and .gitattributes outlive any one build.

    A control: it holds before this change, when nothing is deleted at all,
    and it must still hold after it.
    """
    monkeypatch.setenv("HF_TOKEN", "t")
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    assert publish_corpus.main(_argv(_stage(tmp_path))) == 0

    patterns = mock_api.upload_folder.call_args.kwargs.get("delete_patterns") or []
    for kept in ("README.md", ".gitattributes", "manifest.json"):
        assert not any(_matches(pattern, kept) for pattern in patterns), kept


@patch("huggingface_hub.HfApi")
def test_a_dry_run_still_uploads_nothing(mock_api_cls, tmp_path):
    """The control: pruning must not turn a dry run into a write."""
    assert publish_corpus.main(_argv(_stage(tmp_path), "--dry-run")) == 0
    mock_api_cls.assert_not_called()


def _matches(pattern: str, path: str) -> bool:
    """Match the way the hub does: `*` does not cross a path separator."""
    import re
    regex = "".join(f"[^/]*" if part == "*" else re.escape(part)
                    for part in re.split(r"(\*)", pattern))
    return re.fullmatch(regex, path) is not None
