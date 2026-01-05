#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.tree._SupplementaryTree."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.contents._SupplementaryContents import (
    SupplementaryContents,
)
from scitex.writer.dataclasses.core._DocumentSection import DocumentSection
from scitex.writer.dataclasses.tree._SupplementaryTree import SupplementaryTree


class TestSupplementaryTreeCreation:
    """Tests for SupplementaryTree instantiation."""

    def test_creates_with_root_path(self, tmp_path):
        """Verify SupplementaryTree creates with root path."""
        tree = SupplementaryTree(root=tmp_path)
        assert tree.root == tmp_path

    def test_git_root_optional(self, tmp_path):
        """Verify git_root defaults to None."""
        tree = SupplementaryTree(root=tmp_path)
        assert tree.git_root is None


class TestSupplementaryTreePostInit:
    """Tests for SupplementaryTree __post_init__ initialization."""

    def test_contents_initialized(self, tmp_path):
        """Verify contents SupplementaryContents is initialized."""
        tree = SupplementaryTree(root=tmp_path)
        assert isinstance(tree.contents, SupplementaryContents)
        assert tree.contents.root == tmp_path / "contents"

    def test_base_initialized(self, tmp_path):
        """Verify base DocumentSection is initialized."""
        tree = SupplementaryTree(root=tmp_path)
        assert isinstance(tree.base, DocumentSection)
        assert tree.base.path == tmp_path / "base.tex"

    def test_supplementary_initialized(self, tmp_path):
        """Verify supplementary DocumentSection is initialized."""
        tree = SupplementaryTree(root=tmp_path)
        assert isinstance(tree.supplementary, DocumentSection)
        assert tree.supplementary.path == tmp_path / "supplementary.tex"

    def test_supplementary_diff_initialized(self, tmp_path):
        """Verify supplementary_diff DocumentSection is initialized."""
        tree = SupplementaryTree(root=tmp_path)
        assert isinstance(tree.supplementary_diff, DocumentSection)
        assert tree.supplementary_diff.path == tmp_path / "supplementary_diff.tex"

    def test_readme_initialized(self, tmp_path):
        """Verify readme DocumentSection is initialized."""
        tree = SupplementaryTree(root=tmp_path)
        assert isinstance(tree.readme, DocumentSection)
        assert tree.readme.path == tmp_path / "README.md"

    def test_archive_initialized(self, tmp_path):
        """Verify archive path is initialized."""
        tree = SupplementaryTree(root=tmp_path)
        assert tree.archive == tmp_path / "archive"


class TestSupplementaryTreeVerifyStructure:
    """Tests for SupplementaryTree verify_structure method."""

    def test_verify_fails_when_empty(self, tmp_path):
        """Verify returns False when structure is empty."""
        tree = SupplementaryTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is False
        assert len(missing) > 0

    def test_verify_passes_with_complete_structure(self, tmp_path):
        """Verify returns True when structure is complete."""
        contents = tmp_path / "contents"
        contents.mkdir()
        (contents / "figures").mkdir()
        (contents / "tables").mkdir()
        (contents / "latex_styles").mkdir()
        (tmp_path / "base.tex").touch()
        (tmp_path / "supplementary.tex").touch()

        tree = SupplementaryTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is True
        assert len(missing) == 0


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
