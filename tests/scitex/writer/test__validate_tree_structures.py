#!/usr/bin/env python3
"""Tests for scitex.writer._validate_tree_structures."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.writer._validate_tree_structures import (
    ProjectValidationError,
    _validate_tree_structure_base,
    validate_tree_structures,
)


class TestProjectValidationError:
    """Tests for ProjectValidationError exception class."""

    def test_inherits_from_exception(self):
        """Verify ProjectValidationError inherits from Exception."""
        assert issubclass(ProjectValidationError, Exception)

    def test_can_be_raised(self):
        """Verify ProjectValidationError can be raised with message."""
        with pytest.raises(ProjectValidationError, match="test error"):
            raise ProjectValidationError("test error")


class TestValidateTreeStructureBase:
    """Tests for _validate_tree_structure_base function."""

    def test_raises_when_directory_missing(self, tmp_path):
        """Verify raises ProjectValidationError when directory is missing."""
        with pytest.raises(ProjectValidationError, match="Required directory missing"):
            _validate_tree_structure_base(tmp_path, "nonexistent_dir")

    def test_returns_true_when_directory_exists_no_tree_class(self, tmp_path):
        """Verify returns True when directory exists and no tree_class."""
        (tmp_path / "test_dir").mkdir()

        result = _validate_tree_structure_base(tmp_path, "test_dir")

        assert result is True

    def test_calls_tree_class_verify_structure(self, tmp_path):
        """Verify tree_class.verify_structure is called."""
        (tmp_path / "test_dir").mkdir()

        mock_tree_class = MagicMock()
        mock_tree_instance = MagicMock()
        mock_tree_instance.verify_structure.return_value = (True, [])
        mock_tree_class.return_value = mock_tree_instance

        result = _validate_tree_structure_base(
            tmp_path, "test_dir", tree_class=mock_tree_class
        )

        mock_tree_class.assert_called_once()
        mock_tree_instance.verify_structure.assert_called_once()
        assert result is True

    def test_raises_when_structure_invalid(self, tmp_path):
        """Verify raises when verify_structure returns invalid."""
        (tmp_path / "test_dir").mkdir()

        mock_tree_class = MagicMock()
        mock_tree_instance = MagicMock()
        mock_tree_instance.verify_structure.return_value = (
            False,
            ["Missing file", "Missing directory"],
        )
        mock_tree_class.return_value = mock_tree_instance

        with pytest.raises(ProjectValidationError, match="structure invalid"):
            _validate_tree_structure_base(
                tmp_path, "test_dir", tree_class=mock_tree_class
            )


class TestValidateTreeStructures:
    """Tests for validate_tree_structures function."""

    def test_validates_all_tree_types(self, tmp_path):
        """Verify validates all tree structures."""
        # Create minimal structure for all trees
        (tmp_path / "config").mkdir()
        (tmp_path / "00_shared").mkdir()
        (tmp_path / "01_manuscript").mkdir()
        (tmp_path / "02_supplementary").mkdir()
        (tmp_path / "03_revision").mkdir()
        (tmp_path / "scripts").mkdir()

        # Mock all internal validation functions
        with patch(
            "scitex.writer._validate_tree_structures._validate_config_structure",
            return_value=True,
        ), patch(
            "scitex.writer._validate_tree_structures._validate_00_shared_structure",
            return_value=True,
        ), patch(
            "scitex.writer._validate_tree_structures._validate_01_manuscript_structure",
            return_value=True,
        ), patch(
            "scitex.writer._validate_tree_structures._validate_02_supplementary_structure",
            return_value=True,
        ), patch(
            "scitex.writer._validate_tree_structures._validate_03_revision_structure",
            return_value=True,
        ), patch(
            "scitex.writer._validate_tree_structures._validate_scripts_structure",
            return_value=True,
        ):
            result = validate_tree_structures(tmp_path)

            assert result is True

    def test_accepts_path_object(self, tmp_path):
        """Verify accepts Path object."""
        # Create minimal structure
        for dir_name in [
            "config",
            "00_shared",
            "01_manuscript",
            "02_supplementary",
            "03_revision",
            "scripts",
        ]:
            (tmp_path / dir_name).mkdir()

        with patch(
            "scitex.writer._validate_tree_structures._validate_config_structure",
            return_value=True,
        ), patch(
            "scitex.writer._validate_tree_structures._validate_00_shared_structure",
            return_value=True,
        ), patch(
            "scitex.writer._validate_tree_structures._validate_01_manuscript_structure",
            return_value=True,
        ), patch(
            "scitex.writer._validate_tree_structures._validate_02_supplementary_structure",
            return_value=True,
        ), patch(
            "scitex.writer._validate_tree_structures._validate_03_revision_structure",
            return_value=True,
        ), patch(
            "scitex.writer._validate_tree_structures._validate_scripts_structure",
            return_value=True,
        ):
            # Should not raise
            result = validate_tree_structures(Path(tmp_path))

            assert result is True


class TestTreeValidators:
    """Tests for TREE_VALIDATORS configuration."""

    def test_tree_validators_has_expected_keys(self):
        """Verify TREE_VALIDATORS has expected keys."""
        from scitex.writer._validate_tree_structures import TREE_VALIDATORS

        expected_keys = [
            "config",
            "00_shared",
            "01_manuscript",
            "02_supplementary",
            "03_revision",
            "scripts",
        ]
        for key in expected_keys:
            assert key in TREE_VALIDATORS

    def test_tree_validators_have_dir_name_and_tree_class(self):
        """Verify each validator has dir_name and tree_class."""
        from scitex.writer._validate_tree_structures import TREE_VALIDATORS

        for key, validator in TREE_VALIDATORS.items():
            assert "dir_name" in validator
            assert "tree_class" in validator


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
