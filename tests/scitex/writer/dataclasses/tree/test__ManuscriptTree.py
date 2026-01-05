#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.tree._ManuscriptTree."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.contents._ManuscriptContents import ManuscriptContents
from scitex.writer.dataclasses.core._DocumentSection import DocumentSection
from scitex.writer.dataclasses.tree._ManuscriptTree import ManuscriptTree


class TestManuscriptTreeCreation:
    """Tests for ManuscriptTree instantiation."""

    def test_creates_with_root_path(self, tmp_path):
        """Verify ManuscriptTree creates with root path."""
        tree = ManuscriptTree(root=tmp_path)
        assert tree.root == tmp_path

    def test_git_root_optional(self, tmp_path):
        """Verify git_root defaults to None."""
        tree = ManuscriptTree(root=tmp_path)
        assert tree.git_root is None


class TestManuscriptTreePostInit:
    """Tests for ManuscriptTree __post_init__ initialization."""

    def test_contents_initialized(self, tmp_path):
        """Verify contents ManuscriptContents is initialized."""
        tree = ManuscriptTree(root=tmp_path)
        assert isinstance(tree.contents, ManuscriptContents)
        assert tree.contents.root == tmp_path / "contents"

    def test_base_initialized(self, tmp_path):
        """Verify base DocumentSection is initialized."""
        tree = ManuscriptTree(root=tmp_path)
        assert isinstance(tree.base, DocumentSection)
        assert tree.base.path == tmp_path / "base.tex"

    def test_readme_initialized(self, tmp_path):
        """Verify readme DocumentSection is initialized."""
        tree = ManuscriptTree(root=tmp_path)
        assert isinstance(tree.readme, DocumentSection)
        assert tree.readme.path == tmp_path / "README.md"

    def test_archive_initialized(self, tmp_path):
        """Verify archive path is initialized."""
        tree = ManuscriptTree(root=tmp_path)
        assert tree.archive == tmp_path / "archive"


class TestManuscriptTreeVerifyStructure:
    """Tests for ManuscriptTree verify_structure method."""

    def test_verify_fails_when_empty(self, tmp_path):
        """Verify returns False when structure is empty."""
        tree = ManuscriptTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is False
        assert len(missing) > 0

    def test_verify_passes_with_complete_structure(self, tmp_path):
        """Verify returns True when structure is complete."""
        contents = tmp_path / "contents"
        contents.mkdir()
        (contents / "abstract.tex").touch()
        (contents / "introduction.tex").touch()
        (contents / "methods.tex").touch()
        (contents / "results.tex").touch()
        (contents / "discussion.tex").touch()
        (tmp_path / "base.tex").touch()

        tree = ManuscriptTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is True
        assert len(missing) == 0

    def test_verify_includes_contents_issues(self, tmp_path):
        """Verify missing list includes issues from contents."""
        (tmp_path / "base.tex").touch()
        (tmp_path / "contents").mkdir()

        tree = ManuscriptTree(root=tmp_path)
        _, missing = tree.verify_structure()

        assert any("abstract.tex" in m for m in missing)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
