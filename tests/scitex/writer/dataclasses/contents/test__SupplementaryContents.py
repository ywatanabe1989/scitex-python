#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.contents._SupplementaryContents."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.contents._SupplementaryContents import (
    SupplementaryContents,
)
from scitex.writer.dataclasses.core._DocumentSection import DocumentSection


class TestSupplementaryContentsCreation:
    """Tests for SupplementaryContents instantiation."""

    def test_creates_with_root_path(self, tmp_path):
        """Verify SupplementaryContents creates with root path."""
        contents = SupplementaryContents(root=tmp_path)
        assert contents.root == tmp_path

    def test_git_root_optional(self, tmp_path):
        """Verify git_root defaults to None."""
        contents = SupplementaryContents(root=tmp_path)
        assert contents.git_root is None

    def test_git_root_can_be_set(self, tmp_path):
        """Verify git_root can be explicitly set."""
        git_root = tmp_path / "project"
        contents = SupplementaryContents(root=tmp_path, git_root=git_root)
        assert contents.git_root == git_root


class TestSupplementaryContentsPostInit:
    """Tests for SupplementaryContents __post_init__ initialization."""

    def test_methods_initialized(self, tmp_path):
        """Verify methods DocumentSection is initialized."""
        contents = SupplementaryContents(root=tmp_path)
        assert isinstance(contents.methods, DocumentSection)
        assert contents.methods.path == tmp_path / "methods.tex"

    def test_results_initialized(self, tmp_path):
        """Verify results DocumentSection is initialized."""
        contents = SupplementaryContents(root=tmp_path)
        assert isinstance(contents.results, DocumentSection)
        assert contents.results.path == tmp_path / "results.tex"

    def test_metadata_sections_initialized(self, tmp_path):
        """Verify metadata sections are initialized."""
        contents = SupplementaryContents(root=tmp_path)
        assert contents.title.path == tmp_path / "title.tex"
        assert contents.authors.path == tmp_path / "authors.tex"
        assert contents.keywords.path == tmp_path / "keywords.tex"
        assert contents.journal_name.path == tmp_path / "journal_name.tex"

    def test_directory_paths_initialized(self, tmp_path):
        """Verify directory paths are initialized."""
        contents = SupplementaryContents(root=tmp_path)
        assert contents.figures == tmp_path / "figures"
        assert contents.tables == tmp_path / "tables"
        assert contents.latex_styles == tmp_path / "latex_styles"

    def test_bibliography_initialized(self, tmp_path):
        """Verify bibliography DocumentSection is initialized."""
        contents = SupplementaryContents(root=tmp_path)
        assert isinstance(contents.bibliography, DocumentSection)
        assert contents.bibliography.path == tmp_path / "bibliography.bib"

    def test_wordcount_initialized(self, tmp_path):
        """Verify wordcount DocumentSection is initialized."""
        contents = SupplementaryContents(root=tmp_path)
        assert isinstance(contents.wordcount, DocumentSection)
        assert contents.wordcount.path == tmp_path / "wordcount.tex"

    def test_git_root_passed_to_sections(self, tmp_path):
        """Verify git_root is passed to DocumentSection instances."""
        git_root = tmp_path / "project"
        contents = SupplementaryContents(root=tmp_path, git_root=git_root)
        assert contents.methods.git_root == git_root
        assert contents.results.git_root == git_root


class TestSupplementaryContentsVerifyStructure:
    """Tests for SupplementaryContents verify_structure method."""

    def test_verify_fails_when_no_dirs_exist(self, tmp_path):
        """Verify returns False when required directories are missing."""
        contents = SupplementaryContents(root=tmp_path)
        is_valid, issues = contents.verify_structure()

        assert is_valid is False
        assert len(issues) == 3

    def test_verify_fails_with_partial_dirs(self, tmp_path):
        """Verify returns False when only some required directories exist."""
        (tmp_path / "figures").mkdir()

        contents = SupplementaryContents(root=tmp_path)
        is_valid, issues = contents.verify_structure()

        assert is_valid is False
        assert len(issues) == 2

    def test_verify_passes_with_all_required_dirs(self, tmp_path):
        """Verify returns True when all required directories exist."""
        (tmp_path / "figures").mkdir()
        (tmp_path / "tables").mkdir()
        (tmp_path / "latex_styles").mkdir()

        contents = SupplementaryContents(root=tmp_path)
        is_valid, issues = contents.verify_structure()

        assert is_valid is True
        assert len(issues) == 0

    def test_verify_issues_list_contains_dirnames(self, tmp_path):
        """Verify issues list contains descriptive entries."""
        contents = SupplementaryContents(root=tmp_path)
        _, issues = contents.verify_structure()

        assert any("figures" in issue for issue in issues)
        assert any("tables" in issue for issue in issues)
        assert any("latex_styles" in issue for issue in issues)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
