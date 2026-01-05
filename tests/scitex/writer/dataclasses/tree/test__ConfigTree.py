#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.tree._ConfigTree."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.core._DocumentSection import DocumentSection
from scitex.writer.dataclasses.tree._ConfigTree import ConfigTree


class TestConfigTreeCreation:
    """Tests for ConfigTree instantiation."""

    def test_creates_with_root_path(self, tmp_path):
        """Verify ConfigTree creates with root path."""
        tree = ConfigTree(root=tmp_path)
        assert tree.root == tmp_path

    def test_git_root_optional(self, tmp_path):
        """Verify git_root defaults to None."""
        tree = ConfigTree(root=tmp_path)
        assert tree.git_root is None

    def test_git_root_can_be_set(self, tmp_path):
        """Verify git_root can be explicitly set."""
        git_root = tmp_path / "project"
        tree = ConfigTree(root=tmp_path, git_root=git_root)
        assert tree.git_root == git_root


class TestConfigTreePostInit:
    """Tests for ConfigTree __post_init__ initialization."""

    def test_config_manuscript_initialized(self, tmp_path):
        """Verify config_manuscript DocumentSection is initialized."""
        tree = ConfigTree(root=tmp_path)
        assert isinstance(tree.config_manuscript, DocumentSection)
        assert tree.config_manuscript.path == tmp_path / "config_manuscript.yaml"

    def test_config_supplementary_initialized(self, tmp_path):
        """Verify config_supplementary DocumentSection is initialized."""
        tree = ConfigTree(root=tmp_path)
        assert isinstance(tree.config_supplementary, DocumentSection)
        assert tree.config_supplementary.path == tmp_path / "config_supplementary.yaml"

    def test_config_revision_initialized(self, tmp_path):
        """Verify config_revision DocumentSection is initialized."""
        tree = ConfigTree(root=tmp_path)
        assert isinstance(tree.config_revision, DocumentSection)
        assert tree.config_revision.path == tmp_path / "config_revision.yaml"

    def test_load_config_initialized(self, tmp_path):
        """Verify load_config DocumentSection is initialized."""
        tree = ConfigTree(root=tmp_path)
        assert isinstance(tree.load_config, DocumentSection)
        assert tree.load_config.path == tmp_path / "load_config.sh"


class TestConfigTreeVerifyStructure:
    """Tests for ConfigTree verify_structure method."""

    def test_verify_fails_when_no_files_exist(self, tmp_path):
        """Verify returns False when required files are missing."""
        tree = ConfigTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is False
        assert len(missing) == 3

    def test_verify_passes_with_all_required_files(self, tmp_path):
        """Verify returns True when all required files exist."""
        (tmp_path / "config_manuscript.yaml").touch()
        (tmp_path / "config_supplementary.yaml").touch()
        (tmp_path / "config_revision.yaml").touch()

        tree = ConfigTree(root=tmp_path)
        is_valid, missing = tree.verify_structure()

        assert is_valid is True
        assert len(missing) == 0

    def test_verify_load_config_not_required(self, tmp_path):
        """Verify load_config.sh is not required for validation."""
        (tmp_path / "config_manuscript.yaml").touch()
        (tmp_path / "config_supplementary.yaml").touch()
        (tmp_path / "config_revision.yaml").touch()

        tree = ConfigTree(root=tmp_path)
        is_valid, _ = tree.verify_structure()

        assert is_valid is True


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
