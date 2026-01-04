#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.tree._RevisionTree."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.contents._RevisionContents import RevisionContents
from scitex.writer.dataclasses.core._DocumentSection import DocumentSection
from scitex.writer.dataclasses.tree._RevisionTree import RevisionTree


class TestRevisionTreeCreation:
    """Tests for RevisionTree instantiation."""

    def test_creates_with_root_path(self, tmp_path):
        """Verify RevisionTree creates with root path."""
        tree = RevisionTree(root=tmp_path)
        assert tree.root == tmp_path

    def test_git_root_optional(self, tmp_path):
        """Verify git_root defaults to None."""
        tree = RevisionTree(root=tmp_path)
        assert tree.git_root is None


class TestRevisionTreePostInit:
    """Tests for RevisionTree __post_init__ initialization."""

    def test_contents_initialized(self, tmp_path):
        """Verify contents RevisionContents is initialized."""
        tree = RevisionTree(root=tmp_path)
        assert isinstance(tree.contents, RevisionContents)
        assert tree.contents.root == tmp_path / "contents"

    def test_base_initialized(self, tmp_path):
        """Verify base DocumentSection is initialized."""
        tree = RevisionTree(root=tmp_path)
        assert isinstance(tree.base, DocumentSection)
        assert tree.base.path == tmp_path / "base.tex"

    def test_revision_initialized(self, tmp_path):
        """Verify revision DocumentSection is initialized."""
        tree = RevisionTree(root=tmp_path)
        assert isinstance(tree.revision, DocumentSection)
        assert tree.revision.path == tmp_path / "revision.tex"

    def test_readme_initialized(self, tmp_path):
        """Verify readme DocumentSection is initialized."""
        tree = RevisionTree(root=tmp_path)
        assert isinstance(tree.readme, DocumentSection)
        assert tree.readme.path == tmp_path / "README.md"

    def test_directory_paths_initialized(self, tmp_path):
        """Verify directory paths are initialized."""
        tree = RevisionTree(root=tmp_path)
        assert tree.archive == tmp_path / "archive"
        assert tree.docs == tmp_path / "docs"


class TestRevisionTreeVerifyStructure:
    """Tests for RevisionTree verify_structure method."""

    def test_verify_fails_when_empty(self, tmp_path):
        """Verify returns False when structure is empty."""
        tree = RevisionTree(root=tmp_path)
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
        (contents / "editor").mkdir()
        (tmp_path / "base.tex").touch()
        (tmp_path / "revision.tex").touch()

        tree = RevisionTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is True
        assert len(missing) == 0


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
