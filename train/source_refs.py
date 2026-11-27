"""Resolve every git source the corpus reads to the commit it is read at.

`train/sources.yaml` names repositories and branches, never commits. Each
build reads every repository at its branch head, resolved once when the build
starts, and writes the commit it resolved into its manifest. The manifest is
therefore the record that makes a build reproducible: `--from-manifest` reads
each repository at the commit a previous build recorded instead of at its
head.

A repository's branch is `dev` unless its entry names another one. An entry
is a workspace-relative path, or a mapping with `path` and `branch`:

    skill_refs:
      branch: dev
      repos:
        - ovos/skills/ovos-skill-weather
        - {path: vendor/skill-musicassistant, branch: main}

A clone with an `origin` remote is fetched first, so the head is the remote's
and not whatever the clone last saw; `fetch=False` reads the clone's own
remote-tracking ref instead. A clone with no remote is read at its local
branch. A branch that resolves to nothing stops the build.

An explicit `refs:` mapping of path to commit is also read, because that is
what a manifest replays and what a fixture repository in a test names. The
shipped `sources.yaml` carries none.
"""
import subprocess
from pathlib import Path

DEFAULT_BRANCH = "dev"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


def _remote_url(repo: Path) -> str:
    out = _git(repo, "remote", "get-url", "origin")
    return out.stdout.strip() if out.returncode == 0 else ""


def _commit(repo: Path, ref: str) -> str:
    out = _git(repo, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}")
    return out.stdout.strip() if out.returncode == 0 else ""


def head(repo: Path, branch: str, fetch: bool = True) -> str:
    """The commit *branch* points at, fetched from `origin` when there is one."""
    remote = _remote_url(repo)
    if remote and fetch:
        out = _git(repo, "fetch", "-q", "origin",
                   f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
        if out.returncode:
            raise SystemExit(f"[refs] {repo}: fetching {branch} from {remote} "
                             f"failed: {out.stderr.strip()}")
    for ref in (f"refs/remotes/origin/{branch}", f"refs/heads/{branch}"):
        sha = _commit(repo, ref)
        if sha:
            return sha
    raise SystemExit(f"[refs] {repo}: no branch {branch}; name the repository's "
                     f"branch in train/sources.yaml")


def ensure_commit(repo: Path, sha: str) -> str:
    """*sha*, fetched from `origin` if the clone does not carry it yet."""
    if not _commit(repo, sha) and _remote_url(repo):
        _git(repo, "fetch", "-q", "origin", sha)
    if not _commit(repo, sha):
        raise SystemExit(f"[refs] {repo}: commit {sha} is not in the clone and "
                         f"origin did not serve it")
    return sha


def skill_entries(cfg: dict) -> dict:
    """`{path: branch}` for every skill repository `skill_refs` lists."""
    section = cfg.get("skill_refs") or {}
    default = section.get("branch", DEFAULT_BRANCH)
    out = {}
    for item in section.get("repos") or []:
        if isinstance(item, str):
            out[item] = default
        else:
            out[item["path"]] = item.get("branch", default)
    return out


def resolve(cfg: dict, ws: Path, fetch: bool = True, recorded: dict = None,
            git_sources: bool = True) -> dict:
    """Resolve every git source of *cfg* in place and return the record.

    Afterwards `cfg["skill_refs"]["refs"]` maps each skill path to a commit
    (None for a repository the workspace has no clone of, which the readers
    count or refuse as they always have) and every git source carries its
    commit under `revision`. *recorded* is a manifest's `source_refs` block:
    given, every commit comes from it instead of from a branch head.
    *git_sources* False leaves the non-skill git sources unread, for a
    builder that reads only the skills.
    """
    ws = Path(ws).expanduser()
    record = {"skill_refs": {}, "git_sources": {}}
    recorded_skills = (recorded or {}).get("skill_refs") or {}
    recorded_git = (recorded or {}).get("git_sources") or {}
    section = cfg.setdefault("skill_refs", {})
    explicit = dict(section.get("refs") or {})
    entries = skill_entries(cfg)
    if recorded is not None:
        entries = {path: rec.get("branch") for path, rec in recorded_skills.items()}
    refs = {}
    for path in sorted(set(entries) | set(explicit)):
        repo = ws / path
        branch = entries.get(path)
        if recorded is not None:
            # A replay that skips a repository, or reads one the manifest
            # never read, is not the build it claims to reproduce.
            sha = recorded_skills[path].get("sha")
            if not repo.is_dir() and sha:
                raise SystemExit(f"[refs] {path}: the manifest read it at {sha} "
                                 f"and the workspace has no clone of it")
            if repo.is_dir() and not sha:
                raise SystemExit(f"[refs] {path}: the manifest recorded no commit "
                                 f"for it, and the workspace has a clone")
        if not repo.is_dir():
            refs[path] = None
            record["skill_refs"][path] = {"branch": branch, "sha": None,
                                          "remote": None, "missing_clone": True}
            continue
        if recorded is not None:
            sha = ensure_commit(repo, recorded_skills[path]["sha"])
        elif path in explicit:
            sha, branch = explicit[path], None
        else:
            sha = head(repo, branch, fetch)
        refs[path] = sha
        record["skill_refs"][path] = {"branch": branch, "sha": sha,
                                      "remote": _remote_url(repo)}
    section["refs"] = refs
    for src in (cfg.get("git_sources") or []) if git_sources else []:
        repo = ws / src["path"]
        branch = src.get("branch", DEFAULT_BRANCH)
        if not repo.is_dir():
            raise SystemExit(f"[{src['id']}] missing clone: {repo}")
        if recorded is not None:
            if src["id"] not in recorded_git:
                raise SystemExit(f"[refs] {src['id']}: the manifest records no "
                                 f"commit for this source")
            sha = ensure_commit(repo, recorded_git[src["id"]]["sha"])
        elif src.get("revision"):
            sha, branch = src["revision"], None
        else:
            sha = head(repo, branch, fetch)
        src["revision"] = sha
        record["git_sources"][src["id"]] = {"path": src["path"], "branch": branch,
                                            "sha": sha, "remote": _remote_url(repo)}
    return record


def recorded_refs(manifest_path: Path) -> dict:
    """The `source_refs` block of a build manifest."""
    import json
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if "source_refs" not in manifest:
        raise SystemExit(f"[refs] {manifest_path} records no source_refs; it was "
                         f"written by a builder that read pinned commits")
    return manifest["source_refs"]
