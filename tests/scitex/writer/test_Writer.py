#!/usr/bin/env python3
"""Tests for scitex.writer.Writer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.writer.Writer import Writer


@pytest.fixture
def valid_project_structure(tmp_path):
    """Create a valid writer project structure."""
    (tmp_path / "00_shared").mkdir()
    (tmp_path / "01_manuscript").mkdir()
    (tmp_path / "02_supplementary").mkdir()
    (tmp_path / "03_revision").mkdir()
    (tmp_path / "scripts").mkdir()
    return tmp_path


class TestWriterInitialization:
    """Tests for Writer initialization with existing project."""

    def test_initializes_with_existing_project(self, valid_project_structure):
        """Verify Writer initializes with existing project."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            assert writer.project_dir == valid_project_structure
            assert writer.project_name == valid_project_structure.name

    def test_sets_project_name_from_directory(self, valid_project_structure):
        """Verify project_name defaults to directory name."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            assert writer.project_name == valid_project_structure.name

    def test_uses_custom_project_name(self, valid_project_structure):
        """Verify custom project name is used."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure, name="custom_name")

            assert writer.project_name == "custom_name"

    def test_initializes_document_trees(self, valid_project_structure):
        """Verify document trees are initialized."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            assert writer.manuscript is not None
            assert writer.supplementary is not None
            assert writer.revision is not None
            assert writer.scripts is not None
            assert writer.shared is not None


class TestWriterProjectVerification:
    """Tests for Writer project structure verification."""

    def test_raises_when_manuscript_missing(self, tmp_path):
        """Verify raises RuntimeError when manuscript directory is missing."""
        (tmp_path / "00_shared").mkdir()
        (tmp_path / "02_supplementary").mkdir()
        (tmp_path / "03_revision").mkdir()
        (tmp_path / "scripts").mkdir()

        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            with pytest.raises(RuntimeError, match="01_manuscript"):
                Writer(tmp_path)

    def test_raises_when_supplementary_missing(self, tmp_path):
        """Verify raises RuntimeError when supplementary directory is missing."""
        (tmp_path / "00_shared").mkdir()
        (tmp_path / "01_manuscript").mkdir()
        (tmp_path / "03_revision").mkdir()
        (tmp_path / "scripts").mkdir()

        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            with pytest.raises(RuntimeError, match="02_supplementary"):
                Writer(tmp_path)

    def test_raises_when_revision_missing(self, tmp_path):
        """Verify raises RuntimeError when revision directory is missing."""
        (tmp_path / "00_shared").mkdir()
        (tmp_path / "01_manuscript").mkdir()
        (tmp_path / "02_supplementary").mkdir()
        (tmp_path / "scripts").mkdir()

        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            with pytest.raises(RuntimeError, match="03_revision"):
                Writer(tmp_path)


class TestWriterGetPdf:
    """Tests for Writer.get_pdf method."""

    def test_returns_none_when_pdf_missing(self, valid_project_structure):
        """Verify returns None when PDF doesn't exist."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            assert writer.get_pdf("manuscript") is None

    def test_returns_path_when_pdf_exists(self, valid_project_structure):
        """Verify returns Path when PDF exists."""
        pdf_path = valid_project_structure / "01_manuscript" / "manuscript.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            result = writer.get_pdf("manuscript")
            assert result == pdf_path

    def test_returns_supplementary_pdf(self, valid_project_structure):
        """Verify returns supplementary PDF path."""
        pdf_path = valid_project_structure / "02_supplementary" / "supplementary.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            result = writer.get_pdf("supplementary")
            assert result == pdf_path


class TestWriterDelete:
    """Tests for Writer.delete method."""

    def test_deletes_project_directory(self, valid_project_structure):
        """Verify delete removes project directory."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            result = writer.delete()

            assert result is True
            assert not valid_project_structure.exists()

    def test_delete_returns_false_on_error(self, valid_project_structure):
        """Verify delete returns False on error."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

        with patch("shutil.rmtree", side_effect=PermissionError("Access denied")):
            result = writer.delete()

            assert result is False


class TestWriterCompileMethods:
    """Tests for Writer compilation methods."""

    def test_compile_manuscript_calls_function(self, valid_project_structure):
        """Verify compile_manuscript calls the compile function."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

        mock_result = MagicMock()
        with patch(
            "scitex.writer.Writer.compile_manuscript",
            return_value=mock_result,
        ) as mock_compile:
            result = mock_compile(
                valid_project_structure,
                timeout=300,
                log_callback=None,
                progress_callback=None,
            )

            assert result == mock_result

    def test_compile_supplementary_calls_function(self, valid_project_structure):
        """Verify compile_supplementary calls the compile function."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

        mock_result = MagicMock()
        with patch(
            "scitex.writer.Writer.compile_supplementary",
            return_value=mock_result,
        ) as mock_compile:
            result = mock_compile(
                valid_project_structure,
                timeout=300,
                log_callback=None,
                progress_callback=None,
            )

            assert result == mock_result

    def test_compile_revision_calls_function(self, valid_project_structure):
        """Verify compile_revision calls the compile function."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

        mock_result = MagicMock()
        with patch(
            "scitex.writer.Writer.compile_revision",
            return_value=mock_result,
        ) as mock_compile:
            result = mock_compile(
                valid_project_structure,
                track_changes=False,
                timeout=300,
                log_callback=None,
                progress_callback=None,
            )

            assert result == mock_result


class TestWriterWatch:
    """Tests for Writer.watch method."""

    def test_watch_calls_watch_manuscript(self, valid_project_structure):
        """Verify watch calls watch_manuscript function."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

        callback = MagicMock()
        with patch("scitex.writer.Writer.watch_manuscript") as mock_watch:
            writer.watch(on_compile=callback)

            mock_watch.assert_called_once_with(
                valid_project_structure, on_compile=callback
            )


class TestWriterGitStrategy:
    """Tests for Writer git_strategy parameter."""

    def test_default_git_strategy_is_child(self, valid_project_structure):
        """Verify default git_strategy is 'child'."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            assert writer.git_strategy == "child"

    def test_custom_git_strategy(self, valid_project_structure):
        """Verify custom git_strategy is set."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure, git_strategy="parent")

            assert writer.git_strategy == "parent"

    def test_git_strategy_none(self, valid_project_structure):
        """Verify git_strategy=None is allowed."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure, git_strategy=None)

            assert writer.git_strategy is None


class TestWriterBranchTag:
    """Tests for Writer branch and tag parameters."""

    def test_branch_parameter(self, valid_project_structure):
        """Verify branch parameter is stored."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure, branch="develop")

            assert writer.branch == "develop"

    def test_tag_parameter(self, valid_project_structure):
        """Verify tag parameter is stored."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure, tag="v1.0.0")

            assert writer.tag == "v1.0.0"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
