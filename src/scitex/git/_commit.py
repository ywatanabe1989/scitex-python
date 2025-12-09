#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/git/commit.py

"""
Git commit operations.

Examples
--------
>>> from pathlib import Path
>>> from scitex.git import git_commit, git_add_all
>>> repo = Path("/path/to/repo")
>>> result = git_add_all(repo)
>>> if result:
...     git_commit(repo, "Initial commit")
"""

from pathlib import Path
from scitex.logging import getLogger
from scitex.sh import sh
from ._utils import _in_directory
from ._constants import EXIT_SUCCESS, EXIT_FAILURE
from ._validation import validate_commit_message, validate_path

logger = getLogger(__name__)


def git_add_all(repo_path: Path, verbose: bool = True) -> bool:
    """
    Add all files to git staging.

    Parameters
    ----------
    repo_path : Path
        Git repository path
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True if successful, False if:
        - repo_path is not a git repository
        - git add command fails
        - path is invalid

    Examples
    --------
    >>> git_add_all(Path("/my/repo"))
    True
    """
    valid, error = validate_path(repo_path, must_exist=True)
    if not valid:
        logger.error(error)
        return False

    if not (repo_path / ".git").exists():
        logger.error(f"Not a git repository: {repo_path}")
        return False

    with _in_directory(repo_path):
        result = sh(["git", "add", "."], verbose=verbose, return_as="dict")

        if not result["success"]:
            error_msg = (
                result["stderr"].strip() if result["stderr"] else "Unknown error"
            )
            logger.error(f"Failed to add files to {repo_path}: {error_msg}")
            return False

        return True


def git_commit(repo_path: Path, message: str, verbose: bool = True) -> bool:
    """
    Create a git commit.

    Parameters
    ----------
    repo_path : Path
        Git repository path
    message : str
        Commit message
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True if successful, False if:
        - repo_path is not a git repository
        - message is empty
        - git commit command fails
        - no changes to commit

    Examples
    --------
    >>> git_commit(Path("/my/repo"), "Fix bug in parser")
    True
    """
    valid, error = validate_path(repo_path, must_exist=True)
    if not valid:
        logger.error(error)
        return False

    if not (repo_path / ".git").exists():
        logger.error(f"Not a git repository: {repo_path}")
        return False

    valid, error = validate_commit_message(message)
    if not valid:
        logger.error(error)
        return False

    with _in_directory(repo_path):
        result = sh(["git", "commit", "-m", message], verbose=verbose, return_as="dict")

        if not result["success"]:
            error_msg = (
                result["stderr"].strip() if result["stderr"] else "Unknown error"
            )
            logger.error(f"Failed to commit in {repo_path}: {error_msg}")
            return False

        if verbose:
            logger.info("Commit created successfully")
        return True


def main(args):
    if args.action == "add":
        success = git_add_all(args.repo_path, args.verbose)
        return EXIT_SUCCESS if success else EXIT_FAILURE
    elif args.action == "commit":
        if not args.message:
            logger.error("Message required for commit action")
            return EXIT_FAILURE
        success = git_commit(args.repo_path, args.message, args.verbose)
        return EXIT_SUCCESS if success else EXIT_FAILURE


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--action", choices=["add", "commit"], required=True)
    parser.add_argument("--message", help="Commit message")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_session():
    """Initialize scitex framework, run main function, and cleanup."""
    from ._session import run_with_session

    run_with_session(parse_args, main)


__all__ = [
    "git_add_all",
    "git_commit",
]


if __name__ == "__main__":
    run_session()

# EOF
