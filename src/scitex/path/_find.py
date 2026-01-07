#!/usr/bin/env python3
# Timestamp: "2026-01-08 02:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/path/_find.py

"""File and directory finding utilities."""

import fnmatch
from pathlib import Path
from typing import List, Optional, Union


def find_git_root() -> Path:
    """Find the root directory of the current git repository.

    Returns
    -------
    Path
        Path to the git repository root.
    """
    import git

    repo = git.Repo(".", search_parent_directories=True)
    return Path(repo.working_tree_dir)


def find_dir(root_dir: Union[str, Path], exp: Union[str, List[str]]) -> List[Path]:
    """Find directories matching pattern."""
    return _find(root_dir, type="d", exp=exp)


def find_file(root_dir: Union[str, Path], exp: Union[str, List[str]]) -> List[Path]:
    """Find files matching pattern."""
    return _find(root_dir, type="f", exp=exp)


def _find(
    rootdir: Union[str, Path],
    type: Optional[str] = "f",
    exp: Union[str, List[str]] = "*",
) -> List[Path]:
    """Mimics the Unix find command.

    Parameters
    ----------
    rootdir : str or Path
        Root directory to search in.
    type : str, optional
        'f' for files, 'd' for directories, None for both.
    exp : str or list of str
        Pattern(s) to match.

    Returns
    -------
    list of Path
        Matching paths.

    Example
    -------
    >>> _find('/path/to/search', "f", "*.txt")
    """
    rootdir = Path(rootdir)
    if isinstance(exp, str):
        exp = [exp]

    exclude_keys = ["/lib/", "/env/", "/build/"]
    matches = []

    for _exp in exp:
        for path in rootdir.rglob("*"):
            # Check type
            if type == "f" and not path.is_file():
                continue
            if type == "d" and not path.is_dir():
                continue

            # Check pattern match
            if _exp and not fnmatch.fnmatch(path.name, _exp):
                continue

            # Check exclusions
            path_str = str(path)
            if any(ek in path_str for ek in exclude_keys):
                continue

            matches.append(path)

    return matches


# EOF
