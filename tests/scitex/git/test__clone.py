#!/usr/bin/env python3

"""Tests for git clone operations."""

from pathlib import Path

import pytest

pytest.importorskip("git")

from scitex.git._clone import clone_repo, git_init


class TestGitInit:
    """Tests for git_init function."""

    def test_git_init_success(self, tmp_path):
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        result = git_init(repo_path, verbose=False)
        assert result is True
        assert (repo_path / ".git").exists()

    def test_git_init_already_initialized(self, tmp_path):
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        git_init(repo_path, verbose=False)
        result = git_init(repo_path, verbose=False)
        assert result is False

    def test_git_init_creates_main_branch(self, tmp_path):
        from git import Repo

        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        repo = Repo(repo_path)
        repo.index.add(["test.txt"])
        repo.index.commit("Initial commit")

        assert repo.active_branch.name == "main"


class TestCloneRepo:
    """Tests for clone_repo function."""

    def test_clone_invalid_url(self, tmp_path):
        target_path = tmp_path / "cloned"
        result = clone_repo("invalid-url", target_path, verbose=False)
        assert result is False

    def test_clone_invalid_url_format(self, tmp_path):
        target_path = tmp_path / "cloned"
        result = clone_repo(
            "https://invalid.com/user/repo.git", target_path, verbose=False
        )
        assert result is False

    def test_clone_branch_and_tag_mutually_exclusive(self, tmp_path):
        """Specifying both branch and tag raises ValueError."""
        target_path = tmp_path / "cloned"
        url = "https://github.com/user/repo.git"

        with pytest.raises(ValueError) as exc_info:
            clone_repo(url, target_path, branch="main", tag="v1.0.0", verbose=False)

        assert "mutually exclusive" in str(exc_info.value).lower()

    def test_clone_with_branch_only(self, tmp_path):
        """Test clone_repo with branch parameter (will fail due to no real repo)."""
        target_path = tmp_path / "cloned"
        url = "https://github.com/nonexistent/repo.git"

        result = clone_repo(url, target_path, branch="develop", verbose=False)
        assert result is False

    def test_clone_with_tag_only(self, tmp_path):
        """Test clone_repo with tag parameter (will fail due to no real repo)."""
        target_path = tmp_path / "cloned"
        url = "https://github.com/nonexistent/repo.git"

        result = clone_repo(url, target_path, tag="v1.0.0", verbose=False)
        assert result is False

    def test_clone_without_branch_or_tag(self, tmp_path):
        """Test clone_repo without branch or tag (will fail due to no real repo)."""
        target_path = tmp_path / "cloned"
        url = "https://github.com/nonexistent/repo.git"

        result = clone_repo(url, target_path, verbose=False)
        assert result is False


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_clone.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/clone.py
#
# """
# Git clone operations.
# """
#
# from pathlib import Path
# from scitex.logging import getLogger
# from scitex.sh import sh
# from ._constants import EXIT_SUCCESS, EXIT_FAILURE
#
# logger = getLogger(__name__)
#
#
# def clone_repo(
#     url: str,
#     target_path: Path,
#     branch: str = None,
#     tag: str = None,
#     verbose: bool = True,
# ) -> bool:
#     """
#     Safely clone a git repository.
#
#     Parameters
#     ----------
#     url : str
#         Git repository URL
#     target_path : Path
#         Destination path for cloning
#     branch : str, optional
#         Specific branch to clone. If None, clones the default branch.
#         Mutually exclusive with tag parameter.
#     tag : str, optional
#         Specific tag/release to clone. If None, clones the default branch.
#         Mutually exclusive with branch parameter.
#     verbose : bool
#         Enable verbose output
#
#     Returns
#     -------
#     bool
#         True if successful, False otherwise
#
#     Raises
#     ------
#     ValueError
#         If both branch and tag are specified
#     """
#     from ._remote import _validate_git_url
#
#     # Validate mutual exclusivity
#     if branch and tag:
#         raise ValueError(
#             "Cannot specify both 'branch' and 'tag' parameters. They are mutually exclusive."
#         )
#
#     if not _validate_git_url(url):
#         logger.error(f"Invalid git URL: {url}")
#         return False
#
#     cmd = ["git", "clone"]
#     ref_info = ""
#     if branch:
#         cmd.extend(["--branch", branch])
#         ref_info = f" (branch: {branch})"
#     elif tag:
#         cmd.extend(["--branch", tag])
#         ref_info = f" (tag: {tag})"
#     cmd.extend([url, str(target_path)])
#
#     result = sh(cmd, verbose=verbose, return_as="dict")
#
#     if not result["success"]:
#         logger.error(f"Failed to clone repository: {result['stderr']}")
#         return False
#
#     if verbose:
#         logger.info(f"Repository cloned successfully{ref_info}")
#     return True
#
#
# def git_init(repo_path: Path, verbose: bool = True) -> bool:
#     """
#     Initialize a new git repository.
#
#     Parameters
#     ----------
#     repo_path : Path
#         Path to initialize as git repository
#     verbose : bool
#         Enable verbose output
#
#     Returns
#     -------
#     bool
#         True if successful, False otherwise
#     """
#     from ._utils import _in_directory
#
#     if (repo_path / ".git").exists():
#         logger.warning("Git repository already initialized")
#         return False
#
#     with _in_directory(repo_path):
#         result = sh(["git", "init", "-b", "main"], verbose=verbose, return_as="dict")
#
#         if not result["success"]:
#             logger.warning(f"Failed to initialize git repository: {result['stderr']}")
#             return False
#
#         if verbose:
#             logger.info("Git repository initialized")
#         return True
#
#
# def main(args):
#     if args.action == "clone":
#         if not args.url:
#             logger.error("URL required for clone action")
#             return EXIT_FAILURE
#         success = clone_repo(
#             args.url, args.path, branch=args.branch, tag=args.tag, verbose=args.verbose
#         )
#         return EXIT_SUCCESS if success else EXIT_FAILURE
#     elif args.action == "init":
#         success = git_init(args.path, args.verbose)
#         return EXIT_SUCCESS if success else EXIT_FAILURE
#
#
# def parse_args():
#     """Parse command line arguments."""
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--action", choices=["clone", "init"], required=True)
#     parser.add_argument("--url", help="Repository URL for cloning")
#     parser.add_argument("--path", type=Path, required=True)
#     parser.add_argument(
#         "--branch", help="Branch to clone (mutually exclusive with --tag)"
#     )
#     parser.add_argument(
#         "--tag", help="Tag/release to clone (mutually exclusive with --branch)"
#     )
#     parser.add_argument("--verbose", action="store_true")
#     return parser.parse_args()
#
#
# def run_session():
#     """Initialize scitex framework, run main function, and cleanup."""
#     from ._session import run_with_session
#
#     run_with_session(parse_args, main)
#
#
# __all__ = [
#     "clone_repo",
#     "git_init",
# ]
#
#
# if __name__ == "__main__":
#     run_session()
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_clone.py
# --------------------------------------------------------------------------------
