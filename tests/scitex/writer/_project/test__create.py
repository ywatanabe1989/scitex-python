#!/usr/bin/env python3
"""Tests for scitex.writer._project._create."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.writer._project._create import ensure_project_exists


class TestEnsureProjectExistsExisting:
    """Tests for ensure_project_exists when project already exists."""

    def test_returns_existing_directory(self, tmp_path):
        """Verify returns existing project directory."""
        project_dir = tmp_path / "my_paper"
        project_dir.mkdir()

        result = ensure_project_exists(project_dir, "my_paper")

        assert result == project_dir

    def test_does_not_call_clone_for_existing(self, tmp_path):
        """Verify clone is not called when project exists."""
        project_dir = tmp_path / "my_paper"
        project_dir.mkdir()

        with patch(
            "scitex.writer._project._create._clone_writer_project"
        ) as mock_clone:
            ensure_project_exists(project_dir, "my_paper")

            mock_clone.assert_not_called()

    def test_existing_directory_with_contents(self, tmp_path):
        """Verify returns existing directory with contents."""
        project_dir = tmp_path / "my_paper"
        project_dir.mkdir()
        (project_dir / "01_manuscript").mkdir()
        (project_dir / "file.tex").write_text("content")

        result = ensure_project_exists(project_dir, "my_paper")

        assert result == project_dir
        assert (result / "01_manuscript").exists()


class TestEnsureProjectExistsNew:
    """Tests for ensure_project_exists when creating new project."""

    def test_calls_clone_with_correct_args(self, tmp_path):
        """Verify clone is called with correct arguments."""
        project_dir = tmp_path / "new_paper"

        def mock_clone_side_effect(*args, **kwargs):
            # Simulate clone creating the directory
            project_dir.mkdir(parents=True, exist_ok=True)
            return True

        with patch(
            "scitex.writer._project._create._clone_writer_project",
            side_effect=mock_clone_side_effect,
        ) as mock_clone:
            ensure_project_exists(project_dir, "new_paper")

            mock_clone.assert_called_once_with(str(project_dir), "child", None, None)

    def test_passes_git_strategy(self, tmp_path):
        """Verify git_strategy is passed to clone."""
        project_dir = tmp_path / "new_paper"

        def mock_clone_side_effect(*args, **kwargs):
            project_dir.mkdir(parents=True, exist_ok=True)
            return True

        with patch(
            "scitex.writer._project._create._clone_writer_project",
            side_effect=mock_clone_side_effect,
        ) as mock_clone:
            ensure_project_exists(project_dir, "new_paper", git_strategy="standalone")

            mock_clone.assert_called_once_with(
                str(project_dir), "standalone", None, None
            )

    def test_passes_branch(self, tmp_path):
        """Verify branch parameter is passed to clone."""
        project_dir = tmp_path / "new_paper"

        def mock_clone_side_effect(*args, **kwargs):
            project_dir.mkdir(parents=True, exist_ok=True)
            return True

        with patch(
            "scitex.writer._project._create._clone_writer_project",
            side_effect=mock_clone_side_effect,
        ) as mock_clone:
            ensure_project_exists(project_dir, "new_paper", branch="develop")

            mock_clone.assert_called_once_with(
                str(project_dir), "child", "develop", None
            )

    def test_passes_tag(self, tmp_path):
        """Verify tag parameter is passed to clone."""
        project_dir = tmp_path / "new_paper"

        def mock_clone_side_effect(*args, **kwargs):
            project_dir.mkdir(parents=True, exist_ok=True)
            return True

        with patch(
            "scitex.writer._project._create._clone_writer_project",
            side_effect=mock_clone_side_effect,
        ) as mock_clone:
            ensure_project_exists(project_dir, "new_paper", tag="v1.0.0")

            mock_clone.assert_called_once_with(
                str(project_dir), "child", None, "v1.0.0"
            )

    def test_returns_created_directory(self, tmp_path):
        """Verify returns the created project directory."""
        project_dir = tmp_path / "new_paper"

        def mock_clone_side_effect(*args, **kwargs):
            project_dir.mkdir(parents=True, exist_ok=True)
            return True

        with patch(
            "scitex.writer._project._create._clone_writer_project",
            side_effect=mock_clone_side_effect,
        ):
            result = ensure_project_exists(project_dir, "new_paper")

            assert result == project_dir


class TestEnsureProjectExistsFailure:
    """Tests for ensure_project_exists failure cases."""

    def test_raises_when_clone_fails(self, tmp_path):
        """Verify raises RuntimeError when clone returns False."""
        project_dir = tmp_path / "new_paper"

        with patch(
            "scitex.writer._project._create._clone_writer_project"
        ) as mock_clone:
            mock_clone.return_value = False

            with pytest.raises(RuntimeError, match="Could not create"):
                ensure_project_exists(project_dir, "new_paper")

    def test_raises_when_directory_not_created(self, tmp_path):
        """Verify raises RuntimeError when directory not created after clone."""
        project_dir = tmp_path / "new_paper"

        with patch(
            "scitex.writer._project._create._clone_writer_project"
        ) as mock_clone:
            mock_clone.return_value = True
            # Don't create directory - simulate clone not creating it

            with pytest.raises(RuntimeError, match="was not created"):
                ensure_project_exists(project_dir, "new_paper")


class TestEnsureProjectExistsGitStrategy:
    """Tests for ensure_project_exists git_strategy parameter."""

    def test_git_strategy_none(self, tmp_path):
        """Verify git_strategy=None is passed correctly."""
        project_dir = tmp_path / "new_paper"

        def mock_clone_side_effect(*args, **kwargs):
            project_dir.mkdir(parents=True, exist_ok=True)
            return True

        with patch(
            "scitex.writer._project._create._clone_writer_project",
            side_effect=mock_clone_side_effect,
        ) as mock_clone:
            ensure_project_exists(project_dir, "new_paper", git_strategy=None)

            mock_clone.assert_called_once_with(str(project_dir), None, None, None)

    def test_default_git_strategy_is_child(self, tmp_path):
        """Verify default git_strategy is 'child'."""
        project_dir = tmp_path / "new_paper"

        def mock_clone_side_effect(*args, **kwargs):
            project_dir.mkdir(parents=True, exist_ok=True)
            return True

        with patch(
            "scitex.writer._project._create._clone_writer_project",
            side_effect=mock_clone_side_effect,
        ) as mock_clone:
            ensure_project_exists(project_dir, "new_paper")

            call_args = mock_clone.call_args[0]
            assert call_args[1] == "child"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
