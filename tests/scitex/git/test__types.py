#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/git/test__types.py

"""Tests for git result types."""

import pytest
pytest.importorskip("git")

from scitex.git._types import (
    BranchResult,
    CloneResult,
    CommitResult,
    GitResult,
)


class TestGitResult:
    def test_create_success_result(self):
        result = GitResult(success=True, message="Operation succeeded")
        assert result.success is True
        assert result.message == "Operation succeeded"
        assert result.stdout is None
        assert result.stderr is None
        assert result.exit_code is None

    def test_create_failure_result(self):
        result = GitResult(
            success=False,
            message="Operation failed",
            stderr="fatal: error message",
            exit_code=1
        )
        assert result.success is False
        assert result.message == "Operation failed"
        assert result.stderr == "fatal: error message"
        assert result.exit_code == 1

    def test_create_with_stdout(self):
        result = GitResult(
            success=True,
            message="Success",
            stdout="command output",
            exit_code=0
        )
        assert result.stdout == "command output"


class TestCommitResult:
    def test_create_success_with_hash(self):
        result = CommitResult(
            success=True,
            message="Commit created",
            commit_hash="abc123def456"
        )
        assert result.success is True
        assert result.commit_hash == "abc123def456"
        assert isinstance(result, GitResult)

    def test_create_failure_no_hash(self):
        result = CommitResult(
            success=False,
            message="Nothing to commit"
        )
        assert result.success is False
        assert result.commit_hash is None


class TestCloneResult:
    def test_create_success_with_path(self):
        result = CloneResult(
            success=True,
            message="Repository cloned",
            repo_path="/path/to/repo"
        )
        assert result.success is True
        assert result.repo_path == "/path/to/repo"
        assert isinstance(result, GitResult)

    def test_create_failure_no_path(self):
        result = CloneResult(
            success=False,
            message="Clone failed",
            stderr="fatal: repository not found"
        )
        assert result.success is False
        assert result.repo_path is None
        assert result.stderr == "fatal: repository not found"


class TestBranchResult:
    def test_create_success_with_name(self):
        result = BranchResult(
            success=True,
            message="Branch created",
            branch_name="feature/new-feature"
        )
        assert result.success is True
        assert result.branch_name == "feature/new-feature"
        assert isinstance(result, GitResult)

    def test_create_failure_no_name(self):
        result = BranchResult(
            success=False,
            message="Branch operation failed"
        )
        assert result.success is False
        assert result.branch_name is None


class TestResultInheritance:
    def test_commit_result_is_git_result(self):
        result = CommitResult(success=True)
        assert isinstance(result, GitResult)

    def test_clone_result_is_git_result(self):
        result = CloneResult(success=True)
        assert isinstance(result, GitResult)

    def test_branch_result_is_git_result(self):
        result = BranchResult(success=True)
        assert isinstance(result, GitResult)


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_types.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/types.py
# 
# """
# Git operation result types.
# 
# This module provides dataclasses for rich error handling and result tracking
# in git operations, addressing the "boolean blindness" issue identified in EVAL.md.
# """
# 
# from dataclasses import dataclass
# from typing import Optional
# 
# 
# @dataclass
# class GitResult:
#     """
#     Result of a git operation.
# 
#     Attributes
#     ----------
#     success : bool
#         Whether the operation succeeded
#     message : Optional[str]
#         Human-readable message about the operation
#     stdout : Optional[str]
#         Standard output from git command
#     stderr : Optional[str]
#         Standard error from git command
#     exit_code : Optional[int]
#         Exit code from git command
# 
#     Examples
#     --------
#     >>> result = GitResult(success=True, message="Commit created")
#     >>> if result.success:
#     ...     print(f"Success: {result.message}")
# 
#     >>> result = GitResult(
#     ...     success=False,
#     ...     message="Not a git repository",
#     ...     stderr="fatal: not a git repository"
#     ... )
#     >>> if not result.success:
#     ...     print(f"Error: {result.message}")
#     ...     print(f"Details: {result.stderr}")
#     """
# 
#     success: bool
#     message: Optional[str] = None
#     stdout: Optional[str] = None
#     stderr: Optional[str] = None
#     exit_code: Optional[int] = None
# 
# 
# @dataclass
# class CommitResult(GitResult):
#     """
#     Result of a git commit operation.
# 
#     Attributes
#     ----------
#     commit_hash : Optional[str]
#         SHA hash of the created commit
# 
#     Examples
#     --------
#     >>> result = CommitResult(
#     ...     success=True,
#     ...     message="Commit created",
#     ...     commit_hash="abc123def456"
#     ... )
#     >>> if result.success:
#     ...     print(f"Created commit {result.commit_hash}")
#     """
# 
#     commit_hash: Optional[str] = None
# 
# 
# @dataclass
# class CloneResult(GitResult):
#     """
#     Result of a git clone operation.
# 
#     Attributes
#     ----------
#     repo_path : Optional[str]
#         Path to the cloned repository
# 
#     Examples
#     --------
#     >>> result = CloneResult(
#     ...     success=True,
#     ...     message="Repository cloned",
#     ...     repo_path="/path/to/repo"
#     ... )
#     >>> if result.success:
#     ...     print(f"Cloned to {result.repo_path}")
#     """
# 
#     repo_path: Optional[str] = None
# 
# 
# @dataclass
# class BranchResult(GitResult):
#     """
#     Result of a git branch operation.
# 
#     Attributes
#     ----------
#     branch_name : Optional[str]
#         Name of the branch created or modified
# 
#     Examples
#     --------
#     >>> result = BranchResult(
#     ...     success=True,
#     ...     message="Branch created",
#     ...     branch_name="feature/new-feature"
#     ... )
#     >>> if result.success:
#     ...     print(f"Branch: {result.branch_name}")
#     """
# 
#     branch_name: Optional[str] = None
# 
# 
# __all__ = [
#     "GitResult",
#     "CommitResult",
#     "CloneResult",
#     "BranchResult",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_types.py
# --------------------------------------------------------------------------------
