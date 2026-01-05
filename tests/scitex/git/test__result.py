#!/usr/bin/env python3

"""Tests for git result dataclasses."""

import pytest

pytest.importorskip("git")

from scitex.git._result import (
    BranchResult,
    CommitResult,
    GitResult,
)


class TestGitResult:
    """Tests for GitResult dataclass."""

    def test_create_success_result(self):
        result = GitResult(success=True)
        assert result.success is True
        assert result.error is None
        assert result.stdout is None
        assert result.stderr is None

    def test_create_failure_result(self):
        result = GitResult(
            success=False,
            error="Operation failed",
            stderr="fatal: error message",
        )
        assert result.success is False
        assert result.error == "Operation failed"
        assert result.stderr == "fatal: error message"

    def test_create_with_stdout(self):
        result = GitResult(
            success=True,
            stdout="command output",
        )
        assert result.stdout == "command output"

    def test_create_with_all_fields(self):
        result = GitResult(
            success=True,
            error=None,
            stdout="output",
            stderr="warning",
        )
        assert result.success is True
        assert result.error is None
        assert result.stdout == "output"
        assert result.stderr == "warning"

    def test_dataclass_equality(self):
        result1 = GitResult(success=True, error=None)
        result2 = GitResult(success=True, error=None)
        assert result1 == result2

    def test_dataclass_inequality(self):
        result1 = GitResult(success=True)
        result2 = GitResult(success=False)
        assert result1 != result2


class TestCommitResult:
    """Tests for CommitResult dataclass."""

    def test_create_success_with_hash(self):
        result = CommitResult(
            success=True,
            commit_hash="abc123def456",
        )
        assert result.success is True
        assert result.commit_hash == "abc123def456"

    def test_create_failure_no_hash(self):
        result = CommitResult(
            success=False,
            error="Nothing to commit",
        )
        assert result.success is False
        assert result.commit_hash is None
        assert result.error == "Nothing to commit"

    def test_inherits_from_git_result(self):
        result = CommitResult(success=True)
        assert isinstance(result, GitResult)

    def test_has_git_result_attributes(self):
        result = CommitResult(
            success=True,
            stdout="output",
            stderr="warning",
            commit_hash="abc123",
        )
        assert hasattr(result, "success")
        assert hasattr(result, "error")
        assert hasattr(result, "stdout")
        assert hasattr(result, "stderr")
        assert hasattr(result, "commit_hash")


class TestBranchResult:
    """Tests for BranchResult dataclass."""

    def test_create_success_with_name(self):
        result = BranchResult(
            success=True,
            branch_name="feature/new-feature",
        )
        assert result.success is True
        assert result.branch_name == "feature/new-feature"

    def test_create_failure_no_name(self):
        result = BranchResult(
            success=False,
            error="Branch operation failed",
        )
        assert result.success is False
        assert result.branch_name is None
        assert result.error == "Branch operation failed"

    def test_inherits_from_git_result(self):
        result = BranchResult(success=True)
        assert isinstance(result, GitResult)

    def test_has_git_result_attributes(self):
        result = BranchResult(
            success=True,
            stdout="output",
            stderr="warning",
            branch_name="main",
        )
        assert hasattr(result, "success")
        assert hasattr(result, "error")
        assert hasattr(result, "stdout")
        assert hasattr(result, "stderr")
        assert hasattr(result, "branch_name")


class TestResultInheritance:
    """Tests for result class inheritance."""

    def test_commit_result_is_git_result(self):
        result = CommitResult(success=True)
        assert isinstance(result, GitResult)

    def test_branch_result_is_git_result(self):
        result = BranchResult(success=True)
        assert isinstance(result, GitResult)

    def test_all_results_share_common_interface(self):
        git_result = GitResult(success=True, stdout="out", stderr="err")
        commit_result = CommitResult(success=True, stdout="out", stderr="err")
        branch_result = BranchResult(success=True, stdout="out", stderr="err")

        for result in [git_result, commit_result, branch_result]:
            assert hasattr(result, "success")
            assert hasattr(result, "error")
            assert hasattr(result, "stdout")
            assert hasattr(result, "stderr")


class TestResultUsagePatterns:
    """Tests for common usage patterns."""

    def test_conditional_on_success(self):
        success_result = CommitResult(success=True, commit_hash="abc123")
        failure_result = CommitResult(success=False, error="Failed")

        if success_result.success:
            assert success_result.commit_hash == "abc123"

        if not failure_result.success:
            assert failure_result.error == "Failed"

    def test_result_as_context_data(self):
        result = BranchResult(
            success=True,
            branch_name="develop",
            stdout="Switched to branch 'develop'",
        )

        context = {
            "success": result.success,
            "branch": result.branch_name,
            "output": result.stdout,
        }

        assert context["success"] is True
        assert context["branch"] == "develop"


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
