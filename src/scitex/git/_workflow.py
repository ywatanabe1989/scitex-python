#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/git/workflow.py

"""
Git workflow operations combining multiple git commands.
"""

from pathlib import Path
from ._commit import git_add_all, git_commit
from ._branch import git_branch_rename, git_checkout_new_branch


def setup_branches(repo_path: Path, template_name: str, verbose: bool = True) -> bool:
    """
    Setup standard git branches (main and develop).

    Parameters
    ----------
    repo_path : Path
        Git repository path
    template_name : str
        Template name for initial commit message
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True if successful, False otherwise

    Notes
    -----
    This function attempts to rollback changes if branch operations fail.
    If add or commit fail, no rollback is needed as the repo state is unchanged.
    """
    if not git_add_all(repo_path, verbose=verbose):
        return False

    if not git_commit(
        repo_path, f"Initial commit from {template_name}", verbose=verbose
    ):
        return False

    if not git_branch_rename(repo_path, "main", verbose=verbose):
        _rollback_commit(repo_path, verbose=verbose)
        return False

    if not git_checkout_new_branch(repo_path, "develop", verbose=verbose):
        _rollback_commit(repo_path, verbose=verbose)
        return False

    return True


def _rollback_commit(repo_path: Path, verbose: bool = True) -> None:
    """
    Rollback the last commit to restore clean state.

    Parameters
    ----------
    repo_path : Path
        Git repository path
    verbose : bool
        Enable verbose output
    """
    from scitex.sh import sh
    from ._utils import _in_directory
    from scitex.logging import getLogger

    logger = getLogger(__name__)

    with _in_directory(repo_path):
        result = sh(
            ["git", "reset", "--soft", "HEAD~1"], verbose=verbose, return_as="dict"
        )
        if result["success"]:
            logger.info("Rolled back commit due to workflow failure")
        else:
            logger.warning(f"Failed to rollback commit: {result['stderr']}")


__all__ = [
    "setup_branches",
]

# EOF
