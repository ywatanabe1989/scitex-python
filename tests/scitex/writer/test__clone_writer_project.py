#!/usr/bin/env python3
"""Tests for scitex.writer._clone_writer_project."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.writer._clone_writer_project import clone_writer_project


class TestCloneWriterProjectSuccess:
    """Tests for clone_writer_project success cases."""

    def test_calls_clone_writer_directory(self, tmp_path):
        """Verify clone_writer_project calls clone_writer_directory."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            result = clone_writer_project(project_dir)

            mock_clone.assert_called_once_with(project_dir, "child", None, None)
            assert result is True

    def test_passes_git_strategy(self, tmp_path):
        """Verify git_strategy is passed to clone_writer_directory."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir, git_strategy="parent")

            mock_clone.assert_called_once_with(project_dir, "parent", None, None)

    def test_passes_branch(self, tmp_path):
        """Verify branch is passed to clone_writer_directory."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir, branch="develop")

            mock_clone.assert_called_once_with(project_dir, "child", "develop", None)

    def test_passes_tag(self, tmp_path):
        """Verify tag is passed to clone_writer_directory."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir, tag="v1.0.0")

            mock_clone.assert_called_once_with(project_dir, "child", None, "v1.0.0")

    def test_returns_true_on_success(self, tmp_path):
        """Verify returns True when clone succeeds."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ):
            result = clone_writer_project(project_dir)

            assert result is True


class TestCloneWriterProjectFailure:
    """Tests for clone_writer_project failure cases."""

    def test_returns_false_when_clone_fails(self, tmp_path):
        """Verify returns False when clone_writer_directory returns False."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=False,
        ):
            result = clone_writer_project(project_dir)

            assert result is False

    def test_returns_false_on_exception(self, tmp_path):
        """Verify returns False when exception is raised."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            side_effect=RuntimeError("Clone failed"),
        ):
            result = clone_writer_project(project_dir)

            assert result is False


class TestCloneWriterProjectGitStrategy:
    """Tests for clone_writer_project git_strategy parameter."""

    def test_default_git_strategy_is_child(self, tmp_path):
        """Verify default git_strategy is 'child'."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir)

            call_args = mock_clone.call_args[0]
            assert call_args[1] == "child"

    def test_git_strategy_none(self, tmp_path):
        """Verify git_strategy=None is passed correctly."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir, git_strategy=None)

            mock_clone.assert_called_once_with(project_dir, None, None, None)

    def test_git_strategy_origin(self, tmp_path):
        """Verify git_strategy='origin' is passed correctly."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir, git_strategy="origin")

            mock_clone.assert_called_once_with(project_dir, "origin", None, None)


class TestCloneWriterProjectExport:
    """Tests for module exports."""

    def test_function_is_exported(self):
        """Verify clone_writer_project is importable."""
        from scitex.writer._clone_writer_project import clone_writer_project

        assert callable(clone_writer_project)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
