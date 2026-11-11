"""
Provenance record written into every checkpoint and result JSON: the git commit of the code
(and whether tracked files had uncommitted changes) and the versions of python, numpy, brian2
and brian2cuda. Reads metadata only: nothing is imported, no random stream is touched.
"""
import platform
import subprocess
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _git(*args):
    try:
        return subprocess.run(("git",) + args, cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None                                      # no git, or not a checkout


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def provenance():
    """dict of JSON-safe values; git_commit / git_dirty are None outside a git checkout"""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return dict(git_commit=_git("rev-parse", "HEAD"), git_dirty=None if status is None else bool(status),
                python=platform.python_version(), numpy=_version("numpy"), brian2=_version("brian2"),
                brian2cuda=_version("brian2cuda"))
