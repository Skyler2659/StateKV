"""Low-cost provenance collection without importing model frameworks."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_PACKAGES = (
    "statekv",
    "torch",
    "transformers",
    "numpy",
    "scipy",
    "pandas",
    "PyYAML",
    "mlx",
    "mlx-lm",
    "datasets",
    "pyarrow",
)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(int(block_size)), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(repository_root: Path, arguments: Sequence[str]) -> Optional[bytes]:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=str(Path(repository_root).resolve()),
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_git_provenance(repository_root: Path) -> Dict[str, Any]:
    """Collect commit and a deterministic dirty-worktree fingerprint.

    The dirty hash covers porcelain status (including untracked paths) and the
    tracked binary diff against HEAD. It intentionally does not read untracked
    file contents, which may include multi-gigabyte experiment artifacts.
    """
    root = Path(repository_root).resolve()
    commit = _git_output(root, ["rev-parse", "HEAD"])
    branch = _git_output(root, ["branch", "--show-current"])
    status = _git_output(
        root, ["status", "--porcelain=v1", "--untracked-files=all"]
    )
    tracked_diff = _git_output(root, ["diff", "--binary", "HEAD", "--"])
    status = status or b""
    tracked_diff = tracked_diff or b""
    dirty = bool(status.strip())
    dirty_hash = None
    if dirty:
        digest = hashlib.sha256()
        digest.update(b"statekv-dirty-worktree-v1\0")
        digest.update(status)
        digest.update(b"\0tracked-diff\0")
        digest.update(tracked_diff)
        dirty_hash = digest.hexdigest()
    return {
        "git_commit": commit.decode().strip() if commit else None,
        "git_branch": branch.decode().strip() if branch else None,
        "git_dirty": dirty,
        "dirty_diff_hash": dirty_hash,
        "dirty_diff_hash_scope": "porcelain-status-plus-tracked-binary-diff-v1",
        "untracked_content_hashed": False,
        "git_status": status.decode(errors="replace").splitlines(),
    }


def _package_versions(packages: Iterable[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package in packages:
        name = str(package)
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_runtime_provenance(
    *,
    command: Optional[Sequence[str]] = None,
    working_directory: Optional[Path] = None,
    packages: Iterable[str] = DEFAULT_PACKAGES,
) -> Dict[str, Any]:
    """Collect command, Python, package, platform, and CPU-level hardware data."""
    uname = platform.uname()
    return {
        "command": list(command if command is not None else sys.argv),
        "working_directory": str(
            Path(working_directory if working_directory is not None else Path.cwd())
            .resolve()
        ),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": _package_versions(packages),
        "platform": platform.platform(),
        "hardware": {
            "system": uname.system,
            "release": uname.release,
            "machine": uname.machine,
            "processor": uname.processor or None,
        },
    }
