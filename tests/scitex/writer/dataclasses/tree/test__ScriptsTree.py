#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.tree._ScriptsTree."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.core._DocumentSection import DocumentSection
from scitex.writer.dataclasses.tree._ScriptsTree import ScriptsTree


class TestScriptsTreeCreation:
    """Tests for ScriptsTree instantiation."""

    def test_creates_with_root_path(self, tmp_path):
        """Verify ScriptsTree creates with root path."""
        tree = ScriptsTree(root=tmp_path)
        assert tree.root == tmp_path

    def test_git_root_optional(self, tmp_path):
        """Verify git_root defaults to None."""
        tree = ScriptsTree(root=tmp_path)
        assert tree.git_root is None


class TestScriptsTreePostInit:
    """Tests for ScriptsTree __post_init__ initialization."""

    def test_subdirectories_initialized(self, tmp_path):
        """Verify subdirectory paths are initialized."""
        tree = ScriptsTree(root=tmp_path)
        assert tree.examples == tmp_path / "examples"
        assert tree.installation == tmp_path / "installation"
        assert tree.powershell == tmp_path / "powershell"
        assert tree.python == tmp_path / "python"
        assert tree.shell == tmp_path / "shell"

    def test_compile_manuscript_initialized(self, tmp_path):
        """Verify compile_manuscript DocumentSection is initialized."""
        tree = ScriptsTree(root=tmp_path)
        assert isinstance(tree.compile_manuscript, DocumentSection)
        assert (
            tree.compile_manuscript.path == tmp_path / "shell" / "compile_manuscript.sh"
        )

    def test_compile_supplementary_initialized(self, tmp_path):
        """Verify compile_supplementary DocumentSection is initialized."""
        tree = ScriptsTree(root=tmp_path)
        assert isinstance(tree.compile_supplementary, DocumentSection)
        assert (
            tree.compile_supplementary.path
            == tmp_path / "shell" / "compile_supplementary.sh"
        )

    def test_compile_revision_initialized(self, tmp_path):
        """Verify compile_revision DocumentSection is initialized."""
        tree = ScriptsTree(root=tmp_path)
        assert isinstance(tree.compile_revision, DocumentSection)
        assert tree.compile_revision.path == tmp_path / "shell" / "compile_revision.sh"

    def test_watch_compile_initialized(self, tmp_path):
        """Verify watch_compile DocumentSection is initialized."""
        tree = ScriptsTree(root=tmp_path)
        assert isinstance(tree.watch_compile, DocumentSection)
        assert tree.watch_compile.path == tmp_path / "shell" / "watch_compile.sh"


class TestScriptsTreeVerifyStructure:
    """Tests for ScriptsTree verify_structure method."""

    def test_verify_fails_when_empty(self, tmp_path):
        """Verify returns False when required directories/files are missing."""
        tree = ScriptsTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is False
        assert len(missing) == 5

    def test_verify_passes_with_complete_structure(self, tmp_path):
        """Verify returns True when structure is complete."""
        (tmp_path / "python").mkdir()
        shell_dir = tmp_path / "shell"
        shell_dir.mkdir()
        (shell_dir / "compile_manuscript.sh").touch()
        (shell_dir / "compile_supplementary.sh").touch()
        (shell_dir / "compile_revision.sh").touch()

        tree = ScriptsTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is True
        assert len(missing) == 0

    def test_verify_watch_compile_not_required(self, tmp_path):
        """Verify watch_compile.sh is not required."""
        (tmp_path / "python").mkdir()
        shell_dir = tmp_path / "shell"
        shell_dir.mkdir()
        (shell_dir / "compile_manuscript.sh").touch()
        (shell_dir / "compile_supplementary.sh").touch()
        (shell_dir / "compile_revision.sh").touch()

        tree = ScriptsTree(root=tmp_path)
        is_valid, _ = tree.verify_structure()

        assert is_valid is True


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
