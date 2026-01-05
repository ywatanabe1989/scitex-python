#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.tree._SharedTree."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.core._DocumentSection import DocumentSection
from scitex.writer.dataclasses.tree._SharedTree import SharedTree


class TestSharedTreeCreation:
    """Tests for SharedTree instantiation."""

    def test_creates_with_root_path(self, tmp_path):
        """Verify SharedTree creates with root path."""
        tree = SharedTree(root=tmp_path)
        assert tree.root == tmp_path

    def test_git_root_optional(self, tmp_path):
        """Verify git_root defaults to None."""
        tree = SharedTree(root=tmp_path)
        assert tree.git_root is None


class TestSharedTreePostInit:
    """Tests for SharedTree __post_init__ initialization."""

    def test_metadata_sections_initialized(self, tmp_path):
        """Verify metadata DocumentSections are initialized."""
        tree = SharedTree(root=tmp_path)
        assert tree.authors.path == tmp_path / "authors.tex"
        assert tree.title.path == tmp_path / "title.tex"
        assert tree.keywords.path == tmp_path / "keywords.tex"
        assert tree.journal_name.path == tmp_path / "journal_name.tex"

    def test_bibliography_initialized(self, tmp_path):
        """Verify bibliography DocumentSection is initialized."""
        tree = SharedTree(root=tmp_path)
        assert isinstance(tree.bibliography, DocumentSection)
        assert tree.bibliography.path == tmp_path / "bib_files" / "bibliography.bib"

    def test_directory_paths_initialized(self, tmp_path):
        """Verify directory paths are initialized."""
        tree = SharedTree(root=tmp_path)
        assert tree.bib_files == tmp_path / "bib_files"
        assert tree.latex_styles == tmp_path / "latex_styles"
        assert tree.templates == tmp_path / "templates"


class TestSharedTreeVerifyStructure:
    """Tests for SharedTree verify_structure method."""

    def test_verify_fails_when_empty(self, tmp_path):
        """Verify returns False when required files are missing."""
        tree = SharedTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is False
        assert len(missing) == 5

    def test_verify_passes_with_all_required_files(self, tmp_path):
        """Verify returns True when all required files exist."""
        (tmp_path / "authors.tex").touch()
        (tmp_path / "title.tex").touch()
        (tmp_path / "keywords.tex").touch()
        (tmp_path / "journal_name.tex").touch()
        (tmp_path / "bib_files").mkdir()
        (tmp_path / "bib_files" / "bibliography.bib").touch()

        tree = SharedTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is True
        assert len(missing) == 0


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
