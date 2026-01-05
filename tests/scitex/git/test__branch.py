#!/usr/bin/env python3

"""Tests for git branch operations."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("git")

from scitex.git._branch import git_branch_rename, git_checkout_new_branch


class TestGitBranchRename:
    """Tests for git_branch_rename function."""

    def test_branch_rename_success(self, tmp_path):
        from scitex.git._clone import git_init
        from scitex.git._commit import git_add_all, git_commit

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Initial commit", verbose=False)

        result = git_branch_rename(repo_path, "main", verbose=False)
        assert result is True

    def test_branch_rename_not_git_repo(self, tmp_path):
        non_git_path = tmp_path / "not_a_repo"
        non_git_path.mkdir()

        result = git_branch_rename(non_git_path, "main", verbose=False)
        assert result is False

    def test_branch_rename_nonexistent_path(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"

        result = git_branch_rename(nonexistent, "main", verbose=False)
        assert result is False

    def test_branch_rename_invalid_name_empty(self, tmp_path):
        from scitex.git._clone import git_init
        from scitex.git._commit import git_add_all, git_commit

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Initial commit", verbose=False)

        result = git_branch_rename(repo_path, "", verbose=False)
        assert result is False

    def test_branch_rename_invalid_name_with_space(self, tmp_path):
        from scitex.git._clone import git_init
        from scitex.git._commit import git_add_all, git_commit

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Initial commit", verbose=False)

        result = git_branch_rename(repo_path, "branch name", verbose=False)
        assert result is False

    def test_branch_rename_invalid_name_starts_with_dash(self, tmp_path):
        from scitex.git._clone import git_init
        from scitex.git._commit import git_add_all, git_commit

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Initial commit", verbose=False)

        result = git_branch_rename(repo_path, "-branch", verbose=False)
        assert result is False


class TestGitCheckoutNewBranch:
    """Tests for git_checkout_new_branch function."""

    def test_checkout_new_branch_success(self, tmp_path):
        from scitex.git._clone import git_init
        from scitex.git._commit import git_add_all, git_commit

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Initial commit", verbose=False)

        result = git_checkout_new_branch(repo_path, "feature", verbose=False)
        assert result is True

    def test_checkout_new_branch_not_git_repo(self, tmp_path):
        non_git_path = tmp_path / "not_a_repo"
        non_git_path.mkdir()

        result = git_checkout_new_branch(non_git_path, "feature", verbose=False)
        assert result is False

    def test_checkout_new_branch_nonexistent_path(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"

        result = git_checkout_new_branch(nonexistent, "feature", verbose=False)
        assert result is False

    def test_checkout_new_branch_invalid_name_empty(self, tmp_path):
        from scitex.git._clone import git_init
        from scitex.git._commit import git_add_all, git_commit

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Initial commit", verbose=False)

        result = git_checkout_new_branch(repo_path, "", verbose=False)
        assert result is False

    def test_checkout_new_branch_invalid_name_with_special_char(self, tmp_path):
        from scitex.git._clone import git_init
        from scitex.git._commit import git_add_all, git_commit

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Initial commit", verbose=False)

        result = git_checkout_new_branch(repo_path, "branch~1", verbose=False)
        assert result is False

    def test_checkout_feature_branch_with_slash(self, tmp_path):
        from git import Repo

        from scitex.git._clone import git_init
        from scitex.git._commit import git_add_all, git_commit

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Initial commit", verbose=False)

        result = git_checkout_new_branch(repo_path, "feature/auth", verbose=False)
        assert result is True

        repo = Repo(repo_path)
        assert repo.active_branch.name == "feature/auth"


class TestBranchSwitching:
    """Tests for branch switching workflows."""

    def test_branch_switching_and_commits(self, tmp_path):
        from git import Repo

        from scitex.git._clone import git_init
        from scitex.git._commit import git_add_all, git_commit

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Initial commit", verbose=False)

        git_checkout_new_branch(repo_path, "feature", verbose=False)

        repo = Repo(repo_path)
        assert repo.active_branch.name == "feature"

        (repo_path / "feature.txt").write_text("feature content")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Add feature", verbose=False)

        assert len(list(repo.iter_commits())) == 2

    def test_checkout_duplicate_branch_fails(self, tmp_path):
        from scitex.git._clone import git_init
        from scitex.git._commit import git_add_all, git_commit

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        git_init(repo_path, verbose=False)

        (repo_path / "test.txt").write_text("test")
        git_add_all(repo_path, verbose=False)
        git_commit(repo_path, "Initial commit", verbose=False)

        result1 = git_checkout_new_branch(repo_path, "feature", verbose=False)
        assert result1 is True

        result2 = git_checkout_new_branch(repo_path, "feature", verbose=False)
        assert result2 is False


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_branch.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/branch.py
#
# """
# Git branch operations.
#
# Examples
# --------
# >>> from pathlib import Path
# >>> from scitex.git import git_branch_rename, git_checkout_new_branch
# >>> repo = Path("/path/to/repo")
# >>> git_branch_rename(repo, "main")
# >>> git_checkout_new_branch(repo, "feature/new-feature")
# """
#
# from pathlib import Path
# from scitex.logging import getLogger
# from scitex.sh import sh
# from ._utils import _in_directory
# from ._constants import EXIT_SUCCESS, EXIT_FAILURE
# from ._validation import validate_branch_name, validate_path
#
# logger = getLogger(__name__)
#
#
# def git_branch_rename(repo_path: Path, new_name: str, verbose: bool = True) -> bool:
#     """
#     Rename current branch.
#
#     Parameters
#     ----------
#     repo_path : Path
#         Git repository path
#     new_name : str
#         New branch name
#     verbose : bool
#         Enable verbose output
#
#     Returns
#     -------
#     bool
#         True if successful, False if:
#         - repo_path is not a git repository
#         - new_name is invalid
#         - git branch command fails
#
#     Examples
#     --------
#     >>> git_branch_rename(Path("/my/repo"), "main")
#     True
#     """
#     valid, error = validate_path(repo_path, must_exist=True)
#     if not valid:
#         logger.error(error)
#         return False
#
#     if not (repo_path / ".git").exists():
#         logger.error(f"Not a git repository: {repo_path}")
#         return False
#
#     valid, error = validate_branch_name(new_name)
#     if not valid:
#         logger.error(error)
#         return False
#
#     with _in_directory(repo_path):
#         result = sh(
#             ["git", "branch", "-M", new_name], verbose=verbose, return_as="dict"
#         )
#
#         if not result["success"]:
#             error_msg = (
#                 result["stderr"].strip() if result["stderr"] else "Unknown error"
#             )
#             logger.error(f"Failed to rename branch in {repo_path}: {error_msg}")
#             return False
#
#         if verbose:
#             logger.info(f"Branch renamed to {new_name}")
#         return True
#
#
# def git_checkout_new_branch(
#     repo_path: Path, branch_name: str, verbose: bool = True
# ) -> bool:
#     """
#     Create and checkout a new branch.
#
#     Parameters
#     ----------
#     repo_path : Path
#         Git repository path
#     branch_name : str
#         New branch name
#     verbose : bool
#         Enable verbose output
#
#     Returns
#     -------
#     bool
#         True if successful, False if:
#         - repo_path is not a git repository
#         - branch_name is invalid
#         - git checkout command fails
#
#     Examples
#     --------
#     >>> git_checkout_new_branch(Path("/my/repo"), "feature/auth")
#     True
#     """
#     valid, error = validate_path(repo_path, must_exist=True)
#     if not valid:
#         logger.error(error)
#         return False
#
#     if not (repo_path / ".git").exists():
#         logger.error(f"Not a git repository: {repo_path}")
#         return False
#
#     valid, error = validate_branch_name(branch_name)
#     if not valid:
#         logger.error(error)
#         return False
#
#     with _in_directory(repo_path):
#         result = sh(
#             ["git", "checkout", "-b", branch_name], verbose=verbose, return_as="dict"
#         )
#
#         if not result["success"]:
#             error_msg = (
#                 result["stderr"].strip() if result["stderr"] else "Unknown error"
#             )
#             logger.error(
#                 f"Failed to create branch {branch_name} in {repo_path}: {error_msg}"
#             )
#             return False
#
#         if verbose:
#             logger.info(f"Switched to new branch: {branch_name}")
#         return True
#
#
# def main(args):
#     if args.action == "rename":
#         success = git_branch_rename(args.repo_path, args.branch_name, args.verbose)
#     elif args.action == "checkout":
#         success = git_checkout_new_branch(
#             args.repo_path, args.branch_name, args.verbose
#         )
#     else:
#         return EXIT_FAILURE
#     return EXIT_SUCCESS if success else EXIT_FAILURE
#
#
# def parse_args():
#     """Parse command line arguments."""
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--repo-path", type=Path, required=True)
#     parser.add_argument("--action", choices=["rename", "checkout"], required=True)
#     parser.add_argument("--branch-name", required=True)
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
#     "git_branch_rename",
#     "git_checkout_new_branch",
# ]
#
#
# if __name__ == "__main__":
#     run_session()
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_branch.py
# --------------------------------------------------------------------------------
