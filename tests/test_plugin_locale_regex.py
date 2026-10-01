"""The `plugin_intents` locale-directory regex must accept 3-letter language
subtags and BCP-47 region/script forms, not just 2-letter `xx`/`xx-yy` --
otherwise a Kabyle (`kab`) locale directory hard-fails the whole build.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_dataset  # noqa: E402


def git_repo(path: Path, files: dict) -> str:
    path.mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True,
                                    env=env, capture_output=True)
    run("init", "-q", "-b", "main")
    for name, text in files.items():
        f = path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "fixture")
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("locale_dir", ["kab", "zsm", "sr-Latn", "es-419"])
def test_3letter_and_bcp47_locale_dirs_are_accepted(tmp_path, locale_dir):
    repo = tmp_path / "plugin"
    rev = git_repo(repo, {f"pkg/locale/{locale_dir}/stop.intent": "stop\nhalt\n"})
    src = {"id": "t", "pipeline_id": "stop", "path": "plugin",
          "revision": rev, "files": "**/locale/*/*.intent"}
    rows, stats = [], {}
    build_dataset.read_plugin_intents(src, tmp_path, rows, stats)
    assert rows, f"{locale_dir} locale directory should have been read"


def test_malformed_locale_dir_still_hard_fails(tmp_path):
    repo = tmp_path / "plugin"
    rev = git_repo(repo, {"pkg/locale/not_a_locale_dir_at_all/stop.intent": "stop\n"})
    src = {"id": "t", "pipeline_id": "stop", "path": "plugin",
          "revision": rev, "files": "**/locale/*/*.intent"}
    with pytest.raises(SystemExit):
        build_dataset.read_plugin_intents(src, tmp_path, [], {})
