#!/usr/bin/env python3
"""Tests for scitex.writer._project._trees."""

from pathlib import Path

import pytest

from scitex.writer._project._trees import create_document_trees
from scitex.writer.dataclasses import ManuscriptTree, RevisionTree, SupplementaryTree
from scitex.writer.dataclasses.tree import ScriptsTree


class TestCreateDocumentTreesReturnTypes:
    """Tests for create_document_trees return types."""

    def test_returns_tuple_of_four(self, tmp_path):
        """Verify returns a tuple of four items."""
        result = create_document_trees(tmp_path, None)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_returns_manuscript_tree(self, tmp_path):
        """Verify first element is ManuscriptTree."""
        manuscript, _, _, _ = create_document_trees(tmp_path, None)

        assert isinstance(manuscript, ManuscriptTree)

    def test_returns_supplementary_tree(self, tmp_path):
        """Verify second element is SupplementaryTree."""
        _, supplementary, _, _ = create_document_trees(tmp_path, None)

        assert isinstance(supplementary, SupplementaryTree)

    def test_returns_revision_tree(self, tmp_path):
        """Verify third element is RevisionTree."""
        _, _, revision, _ = create_document_trees(tmp_path, None)

        assert isinstance(revision, RevisionTree)

    def test_returns_scripts_tree(self, tmp_path):
        """Verify fourth element is ScriptsTree."""
        _, _, _, scripts = create_document_trees(tmp_path, None)

        assert isinstance(scripts, ScriptsTree)


class TestCreateDocumentTreesPaths:
    """Tests for create_document_trees path initialization."""

    def test_manuscript_path_correct(self, tmp_path):
        """Verify manuscript tree has correct root path."""
        manuscript, _, _, _ = create_document_trees(tmp_path, None)

        assert manuscript.root == tmp_path / "01_manuscript"

    def test_supplementary_path_correct(self, tmp_path):
        """Verify supplementary tree has correct root path."""
        _, supplementary, _, _ = create_document_trees(tmp_path, None)

        assert supplementary.root == tmp_path / "02_supplementary"

    def test_revision_path_correct(self, tmp_path):
        """Verify revision tree has correct root path."""
        _, _, revision, _ = create_document_trees(tmp_path, None)

        assert revision.root == tmp_path / "03_revision"

    def test_scripts_path_correct(self, tmp_path):
        """Verify scripts tree has correct root path."""
        _, _, _, scripts = create_document_trees(tmp_path, None)

        assert scripts.root == tmp_path / "scripts"


class TestCreateDocumentTreesGitRoot:
    """Tests for create_document_trees git_root propagation."""

    def test_git_root_none_propagated(self, tmp_path):
        """Verify git_root=None is propagated to all trees."""
        manuscript, supplementary, revision, scripts = create_document_trees(
            tmp_path, None
        )

        assert manuscript.git_root is None
        assert supplementary.git_root is None
        assert revision.git_root is None
        assert scripts.git_root is None

    def test_git_root_propagated_to_all(self, tmp_path):
        """Verify git_root is propagated to all trees."""
        git_root = tmp_path / ".git"
        git_root.mkdir()

        manuscript, supplementary, revision, scripts = create_document_trees(
            tmp_path, git_root
        )

        assert manuscript.git_root == git_root
        assert supplementary.git_root == git_root
        assert revision.git_root == git_root
        assert scripts.git_root == git_root


class TestCreateDocumentTreesNestedProject:
    """Tests for create_document_trees with nested project directory."""

    def test_nested_project_path(self, tmp_path):
        """Verify works with nested project directory."""
        nested = tmp_path / "papers" / "2024" / "my_paper"

        manuscript, supplementary, revision, scripts = create_document_trees(
            nested, None
        )

        assert manuscript.root == nested / "01_manuscript"
        assert supplementary.root == nested / "02_supplementary"
        assert revision.root == nested / "03_revision"
        assert scripts.root == nested / "scripts"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
