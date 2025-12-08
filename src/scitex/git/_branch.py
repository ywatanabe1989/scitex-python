#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/git/branch.py

"""
Git branch operations.

Examples
--------
>>> from pathlib import Path
>>> from scitex.git import git_branch_rename, git_checkout_new_branch
>>> repo = Path("/path/to/repo")
>>> git_branch_rename(repo, "main")
>>> git_checkout_new_branch(repo, "feature/new-feature")
"""

from pathlib import Path
from scitex.logging import getLogger
from scitex.sh import sh
from ._utils import _in_directory
from ._constants import EXIT_SUCCESS, EXIT_FAILURE
from ._validation import validate_branch_name, validate_path

logger = getLogger(__name__)


def git_branch_rename(repo_path: Path, new_name: str, verbose: bool = True) -> bool:
    """
    Rename current branch.

    Parameters
    ----------
    repo_path : Path
        Git repository path
    new_name : str
        New branch name
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True if successful, False if:
        - repo_path is not a git repository
        - new_name is invalid
        - git branch command fails

    Examples
    --------
    >>> git_branch_rename(Path("/my/repo"), "main")
    True
    """
    valid, error = validate_path(repo_path, must_exist=True)
    if not valid:
        logger.error(error)
        return False

    if not (repo_path / ".git").exists():
        logger.error(f"Not a git repository: {repo_path}")
        return False

    valid, error = validate_branch_name(new_name)
    if not valid:
        logger.error(error)
        return False

    with _in_directory(repo_path):
        result = sh(
            ["git", "branch", "-M", new_name], verbose=verbose, return_as="dict"
        )

        if not result["success"]:
            error_msg = (
                result["stderr"].strip() if result["stderr"] else "Unknown error"
            )
            logger.error(f"Failed to rename branch in {repo_path}: {error_msg}")
            return False

        if verbose:
            logger.info(f"Branch renamed to {new_name}")
        return True


def git_checkout_new_branch(
    repo_path: Path, branch_name: str, verbose: bool = True
) -> bool:
    """
    Create and checkout a new branch.

    Parameters
    ----------
    repo_path : Path
        Git repository path
    branch_name : str
        New branch name
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True if successful, False if:
        - repo_path is not a git repository
        - branch_name is invalid
        - git checkout command fails

    Examples
    --------
    >>> git_checkout_new_branch(Path("/my/repo"), "feature/auth")
    True
    """
    valid, error = validate_path(repo_path, must_exist=True)
    if not valid:
        logger.error(error)
        return False

    if not (repo_path / ".git").exists():
        logger.error(f"Not a git repository: {repo_path}")
        return False

    valid, error = validate_branch_name(branch_name)
    if not valid:
        logger.error(error)
        return False

    with _in_directory(repo_path):
        result = sh(
            ["git", "checkout", "-b", branch_name], verbose=verbose, return_as="dict"
        )

        if not result["success"]:
            error_msg = (
                result["stderr"].strip() if result["stderr"] else "Unknown error"
            )
            logger.error(
                f"Failed to create branch {branch_name} in {repo_path}: {error_msg}"
            )
            return False

        if verbose:
            logger.info(f"Switched to new branch: {branch_name}")
        return True


def main(args):
    if args.action == "rename":
        success = git_branch_rename(args.repo_path, args.branch_name, args.verbose)
    elif args.action == "checkout":
        success = git_checkout_new_branch(
            args.repo_path, args.branch_name, args.verbose
        )
    else:
        return EXIT_FAILURE
    return EXIT_SUCCESS if success else EXIT_FAILURE


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--action", choices=["rename", "checkout"], required=True)
    parser.add_argument("--branch-name", required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_session():
    """Initialize scitex framework, run main function, and cleanup."""
    from ._session import run_with_session

    run_with_session(parse_args, main)


__all__ = [
    "git_branch_rename",
    "git_checkout_new_branch",
]


if __name__ == "__main__":
    run_session()

# EOF
