#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/git/test_workflow.py

"""Tests for git workflow operations."""

import tempfile
from pathlib import Path

import pytest
pytest.importorskip("git")

from scitex.git._clone import git_init
from scitex.git._workflow import setup_branches


class TestWorkflow:
    def test_setup_branches_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            git_init(repo_path)

            test_file = repo_path / "test.txt"
            test_file.write_text("test content")

            result = setup_branches(repo_path, "test-template", verbose=False)
            assert result is True

    def test_setup_branches_no_git(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            test_file = repo_path / "test.txt"
            test_file.write_text("test content")

            result = setup_branches(repo_path, "test-template", verbose=False)
            assert result is False

    def test_complete_git_workflow(self):
        from git import Repo

        from scitex.git._branch import git_branch_rename
        from scitex.git._commit import git_add_all, git_commit

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            git_init(repo_path, verbose=False)

            test_file = repo_path / "test.txt"
            test_file.write_text("initial content")

            git_add_all(repo_path, verbose=False)
            git_commit(repo_path, "Initial commit", verbose=False)

            repo = Repo(repo_path)
            assert len(list(repo.iter_commits())) == 1
            assert repo.head.commit.message.strip() == "Initial commit"

            git_branch_rename(repo_path, "main", verbose=False)
            assert repo.active_branch.name == "main"

    def test_multi_file_commit_workflow(self):
        from git import Repo

        from scitex.git._commit import git_add_all, git_commit

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            git_init(repo_path, verbose=False)

            (repo_path / "file1.txt").write_text("content 1")
            (repo_path / "file2.txt").write_text("content 2")
            (repo_path / "file3.txt").write_text("content 3")

            git_add_all(repo_path, verbose=False)
            git_commit(repo_path, "Add multiple files", verbose=False)

            repo = Repo(repo_path)
            commit = repo.head.commit
            assert commit.message.strip() == "Add multiple files"
            assert len(list(commit.tree.traverse())) == 3

# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_workflow.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/workflow.py
# 
# """
# Git workflow operations combining multiple git commands.
# """
# 
# from pathlib import Path
# from ._commit import git_add_all, git_commit
# from ._branch import git_branch_rename, git_checkout_new_branch
# 
# 
# def setup_branches(repo_path: Path, template_name: str, verbose: bool = True) -> bool:
#     """
#     Setup standard git branches (main and develop).
# 
#     Parameters
#     ----------
#     repo_path : Path
#         Git repository path
#     template_name : str
#         Template name for initial commit message
#     verbose : bool
#         Enable verbose output
# 
#     Returns
#     -------
#     bool
#         True if successful, False otherwise
# 
#     Notes
#     -----
#     This function attempts to rollback changes if branch operations fail.
#     If add or commit fail, no rollback is needed as the repo state is unchanged.
#     """
#     if not git_add_all(repo_path, verbose=verbose):
#         return False
# 
#     if not git_commit(
#         repo_path, f"Initial commit from {template_name}", verbose=verbose
#     ):
#         return False
# 
#     if not git_branch_rename(repo_path, "main", verbose=verbose):
#         _rollback_commit(repo_path, verbose=verbose)
#         return False
# 
#     if not git_checkout_new_branch(repo_path, "develop", verbose=verbose):
#         _rollback_commit(repo_path, verbose=verbose)
#         return False
# 
#     return True
# 
# 
# def _rollback_commit(repo_path: Path, verbose: bool = True) -> None:
#     """
#     Rollback the last commit to restore clean state.
# 
#     Parameters
#     ----------
#     repo_path : Path
#         Git repository path
#     verbose : bool
#         Enable verbose output
#     """
#     from scitex.sh import sh
#     from ._utils import _in_directory
#     from scitex.logging import getLogger
# 
#     logger = getLogger(__name__)
# 
#     with _in_directory(repo_path):
#         result = sh(
#             ["git", "reset", "--soft", "HEAD~1"], verbose=verbose, return_as="dict"
#         )
#         if result["success"]:
#             logger.info("Rolled back commit due to workflow failure")
#         else:
#             logger.warning(f"Failed to rollback commit: {result['stderr']}")
# 
# 
# __all__ = [
#     "setup_branches",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_workflow.py
# --------------------------------------------------------------------------------
