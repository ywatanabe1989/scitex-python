#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/git/clone.py

"""
Git clone operations.
"""

from pathlib import Path
from scitex.logging import getLogger
from scitex.sh import sh
from ._constants import EXIT_SUCCESS, EXIT_FAILURE

logger = getLogger(__name__)


def clone_repo(
    url: str,
    target_path: Path,
    branch: str = None,
    tag: str = None,
    verbose: bool = True,
) -> bool:
    """
    Safely clone a git repository.

    Parameters
    ----------
    url : str
        Git repository URL
    target_path : Path
        Destination path for cloning
    branch : str, optional
        Specific branch to clone. If None, clones the default branch.
        Mutually exclusive with tag parameter.
    tag : str, optional
        Specific tag/release to clone. If None, clones the default branch.
        Mutually exclusive with branch parameter.
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True if successful, False otherwise

    Raises
    ------
    ValueError
        If both branch and tag are specified
    """
    from ._remote import _validate_git_url

    # Validate mutual exclusivity
    if branch and tag:
        raise ValueError(
            "Cannot specify both 'branch' and 'tag' parameters. They are mutually exclusive."
        )

    if not _validate_git_url(url):
        logger.error(f"Invalid git URL: {url}")
        return False

    cmd = ["git", "clone"]
    ref_info = ""
    if branch:
        cmd.extend(["--branch", branch])
        ref_info = f" (branch: {branch})"
    elif tag:
        cmd.extend(["--branch", tag])
        ref_info = f" (tag: {tag})"
    cmd.extend([url, str(target_path)])

    result = sh(cmd, verbose=verbose, return_as="dict")

    if not result["success"]:
        logger.error(f"Failed to clone repository: {result['stderr']}")
        return False

    if verbose:
        logger.info(f"Repository cloned successfully{ref_info}")
    return True


def git_init(repo_path: Path, verbose: bool = True) -> bool:
    """
    Initialize a new git repository.

    Parameters
    ----------
    repo_path : Path
        Path to initialize as git repository
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    from ._utils import _in_directory

    if (repo_path / ".git").exists():
        logger.warning("Git repository already initialized")
        return False

    with _in_directory(repo_path):
        result = sh(["git", "init", "-b", "main"], verbose=verbose, return_as="dict")

        if not result["success"]:
            logger.warning(f"Failed to initialize git repository: {result['stderr']}")
            return False

        if verbose:
            logger.info("Git repository initialized")
        return True


def main(args):
    if args.action == "clone":
        if not args.url:
            logger.error("URL required for clone action")
            return EXIT_FAILURE
        success = clone_repo(
            args.url, args.path, branch=args.branch, tag=args.tag, verbose=args.verbose
        )
        return EXIT_SUCCESS if success else EXIT_FAILURE
    elif args.action == "init":
        success = git_init(args.path, args.verbose)
        return EXIT_SUCCESS if success else EXIT_FAILURE


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["clone", "init"], required=True)
    parser.add_argument("--url", help="Repository URL for cloning")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--branch", help="Branch to clone (mutually exclusive with --tag)"
    )
    parser.add_argument(
        "--tag", help="Tag/release to clone (mutually exclusive with --branch)"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_session():
    """Initialize scitex framework, run main function, and cleanup."""
    from ._session import run_with_session

    run_with_session(parse_args, main)


__all__ = [
    "clone_repo",
    "git_init",
]


if __name__ == "__main__":
    run_session()

# EOF
