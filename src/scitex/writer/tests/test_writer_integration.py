#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for Writer class with actual project creation and git operations.

These tests use real temporary directories and git operations.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scitex.writer.Writer import Writer


class TestWriterProjectAttachment:
    """Test Writer attachment to existing projects."""

    @pytest.fixture
    def valid_project_dir(self):
        """Create a valid project structure."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_valid_")
        project_dir = Path(temp_dir)

        # Create required structure
        for dir_name in ["01_manuscript", "02_supplementary", "03_revision"]:
            (project_dir / dir_name).mkdir(parents=True, exist_ok=True)

        yield project_dir

        # Cleanup
        if project_dir.exists():
            shutil.rmtree(project_dir)

    @pytest.fixture
    def invalid_project_dir(self):
        """Create invalid project (missing required directories)."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_invalid_")
        project_dir = Path(temp_dir)

        # Only create one directory (incomplete)
        (project_dir / "01_manuscript").mkdir(parents=True, exist_ok=True)

        yield project_dir

        # Cleanup
        if project_dir.exists():
            shutil.rmtree(project_dir)

    def test_attach_to_valid_project_structure(self, valid_project_dir):
        """Test Writer can attach to valid existing project."""
        # Verify fixture setup
        assert valid_project_dir.exists()
        assert (valid_project_dir / "01_manuscript").exists()
        assert (valid_project_dir / "02_supplementary").exists()
        assert (valid_project_dir / "03_revision").exists()

        # Test: Writer should not raise when attaching to valid structure
        with patch("scitex.git.init_git_repo", return_value=None):
            try:
                writer = Writer(valid_project_dir, git_strategy=None)
                assert writer.project_dir == valid_project_dir
                assert writer.project_name == valid_project_dir.name
            except RuntimeError:
                pytest.fail("Valid project should not raise RuntimeError")

    def test_attach_to_invalid_project_raises_error(self, invalid_project_dir):
        """Test Writer raises error when attaching to invalid project."""
        assert invalid_project_dir.exists()
        assert (invalid_project_dir / "01_manuscript").exists()
        assert not (invalid_project_dir / "02_supplementary").exists()

        # Test: Writer should raise when structure is invalid
        with patch("scitex.git.init_git_repo", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                writer = Writer(invalid_project_dir, git_strategy=None)

            # Verify error message mentions missing directory
            assert "Project structure invalid" in str(exc_info.value) or \
                   "missing" in str(exc_info.value)


class TestProjectStructureVerification:
    """Test _verify_project_structure() method."""

    @pytest.fixture
    def writer_with_valid_project(self):
        """Create Writer instance with valid project."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_verify_")
        project_dir = Path(temp_dir)

        # Create structure
        for dir_name in ["01_manuscript", "02_supplementary", "03_revision"]:
            (project_dir / dir_name).mkdir(parents=True, exist_ok=True)

        with patch("scitex.writer.Writer.ManuscriptTree"), \
             patch("scitex.writer.Writer.SupplementaryTree"), \
             patch("scitex.writer.Writer.RevisionTree"), \
             patch("scitex.git.init_git_repo", return_value=None):
            writer = Writer(project_dir, git_strategy=None)

        yield writer

        # Cleanup
        if project_dir.exists():
            shutil.rmtree(project_dir)

    def test_verify_structure_success(self, writer_with_valid_project):
        """Test structure verification succeeds with all directories."""
        # Should not raise
        try:
            writer_with_valid_project._verify_project_structure()
        except Exception as e:
            pytest.fail(f"Structure verification should not raise: {e}")

    def test_verify_structure_detects_missing_manuscript(self):
        """Test verification detects missing 01_manuscript."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_no_ms_")
        project_dir = Path(temp_dir)

        # Create only supplementary and revision
        (project_dir / "02_supplementary").mkdir(parents=True, exist_ok=True)
        (project_dir / "03_revision").mkdir(parents=True, exist_ok=True)

        with patch("scitex.git.init_git_repo", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                writer = Writer(project_dir, git_strategy=None)

            assert "01_manuscript" in str(exc_info.value)

        # Cleanup
        shutil.rmtree(project_dir)

    def test_verify_structure_detects_missing_supplementary(self):
        """Test verification detects missing 02_supplementary."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_no_supp_")
        project_dir = Path(temp_dir)

        # Create only manuscript and revision
        (project_dir / "01_manuscript").mkdir(parents=True, exist_ok=True)
        (project_dir / "03_revision").mkdir(parents=True, exist_ok=True)

        with patch("scitex.git.init_git_repo", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                writer = Writer(project_dir, git_strategy=None)

            assert "02_supplementary" in str(exc_info.value)

        # Cleanup
        shutil.rmtree(project_dir)

    def test_verify_structure_detects_missing_revision(self):
        """Test verification detects missing 03_revision."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_no_rev_")
        project_dir = Path(temp_dir)

        # Create only manuscript and supplementary
        (project_dir / "01_manuscript").mkdir(parents=True, exist_ok=True)
        (project_dir / "02_supplementary").mkdir(parents=True, exist_ok=True)

        with patch("scitex.git.init_git_repo", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                writer = Writer(project_dir, git_strategy=None)

            assert "03_revision" in str(exc_info.value)

        # Cleanup
        shutil.rmtree(project_dir)


class TestChildGitRemoval:
    """Test child git removal functionality.

    Note: _remove_child_git() method has been refactored.
    Git removal is now handled by scitex.git.init_git_repo() based on git_strategy.
    These tests are kept for reference but skipped.
    """

    @pytest.mark.skip(reason="Method _remove_child_git() refactored into scitex.git module")
    def test_remove_child_git_when_exists(self):
        """Test removal of child .git folder."""
        pass

    @pytest.mark.skip(reason="Method _remove_child_git() refactored into scitex.git module")
    def test_remove_child_git_when_not_exists(self):
        """Test _remove_child_git handles missing .git gracefully."""
        pass


class TestProjectNameHandling:
    """Test project name storage and usage."""

    def test_project_name_from_explicit_parameter(self):
        """Test Writer stores explicit project name parameter."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_name_")
        project_dir = Path(temp_dir)

        # Create structure
        for dir_name in ["01_manuscript", "02_supplementary", "03_revision"]:
            (project_dir / dir_name).mkdir(parents=True, exist_ok=True)

        with patch("scitex.git.init_git_repo", return_value=None):
            with patch("scitex.writer.Writer.ManuscriptTree"), \
                 patch("scitex.writer.Writer.SupplementaryTree"), \
                 patch("scitex.writer.Writer.RevisionTree"):
                writer = Writer(project_dir, name="my_custom_paper", git_strategy=None)

        assert writer.project_name == "my_custom_paper"

        # Cleanup
        shutil.rmtree(project_dir)

    def test_project_name_from_directory(self):
        """Test Writer derives project name from directory if not provided."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_")
        project_dir = Path(temp_dir) / "my_paper"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create structure
        for dir_name in ["01_manuscript", "02_supplementary", "03_revision"]:
            (project_dir / dir_name).mkdir(parents=True, exist_ok=True)

        with patch("scitex.git.init_git_repo", return_value=None):
            with patch("scitex.writer.Writer.ManuscriptTree"), \
                 patch("scitex.writer.Writer.SupplementaryTree"), \
                 patch("scitex.writer.Writer.RevisionTree"):
                writer = Writer(project_dir, git_strategy=None)

        assert writer.project_name == "my_paper"

        # Cleanup
        shutil.rmtree(temp_dir)


class TestWriterAttributes:
    """Test Writer instance attributes."""

    def test_writer_has_project_dir(self):
        """Test Writer stores project_dir."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_attr_")
        project_dir = Path(temp_dir)

        # Create structure
        for dir_name in ["01_manuscript", "02_supplementary", "03_revision"]:
            (project_dir / dir_name).mkdir(parents=True, exist_ok=True)

        with patch("scitex.git.init_git_repo", return_value=None):
            with patch("scitex.writer.Writer.ManuscriptTree"), \
                 patch("scitex.writer.Writer.SupplementaryTree"), \
                 patch("scitex.writer.Writer.RevisionTree"):
                writer = Writer(project_dir, git_strategy=None)

        assert hasattr(writer, "project_dir")
        assert writer.project_dir == project_dir

        # Cleanup
        shutil.rmtree(project_dir)

    def test_writer_has_git_root(self):
        """Test Writer stores git_root."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_git_root_")
        project_dir = Path(temp_dir)

        # Create structure
        for dir_name in ["01_manuscript", "02_supplementary", "03_revision"]:
            (project_dir / dir_name).mkdir(parents=True, exist_ok=True)

        with patch("scitex.git.init_git_repo", return_value=None):
            with patch("scitex.writer.Writer.ManuscriptTree"), \
                 patch("scitex.writer.Writer.SupplementaryTree"), \
                 patch("scitex.writer.Writer.RevisionTree"):
                writer = Writer(project_dir, git_strategy=None)

        assert hasattr(writer, "git_root")

        # Cleanup
        shutil.rmtree(project_dir)

    def test_writer_has_document_trees(self):
        """Test Writer creates document tree attributes."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_trees_")
        project_dir = Path(temp_dir)

        # Create structure
        for dir_name in ["01_manuscript", "02_supplementary", "03_revision"]:
            (project_dir / dir_name).mkdir(parents=True, exist_ok=True)

        with patch("scitex.git.init_git_repo", return_value=None):
            writer = Writer(project_dir, git_strategy=None)

        assert hasattr(writer, "manuscript")
        assert hasattr(writer, "supplementary")
        assert hasattr(writer, "revision")

        # Cleanup
        shutil.rmtree(project_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
