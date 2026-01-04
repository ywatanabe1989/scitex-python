#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.config._WriterConfig."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.config._WriterConfig import WriterConfig


class TestWriterConfigCreation:
    """Tests for WriterConfig instantiation."""

    def test_dataclass_fields(self):
        """Verify WriterConfig has required dataclass fields."""
        config = WriterConfig(
            project_dir=Path("/tmp"),
            manuscript_dir=Path("/tmp/01_manuscript"),
            supplementary_dir=Path("/tmp/02_supplementary"),
            revision_dir=Path("/tmp/03_revision"),
            shared_dir=Path("/tmp/00_shared"),
        )
        assert config.project_dir == Path("/tmp")
        assert config.manuscript_dir == Path("/tmp/01_manuscript")
        assert config.supplementary_dir == Path("/tmp/02_supplementary")
        assert config.revision_dir == Path("/tmp/03_revision")
        assert config.shared_dir == Path("/tmp/00_shared")

    def test_compile_script_optional(self):
        """Verify compile_script defaults to None."""
        config = WriterConfig(
            project_dir=Path("/tmp"),
            manuscript_dir=Path("/tmp/01_manuscript"),
            supplementary_dir=Path("/tmp/02_supplementary"),
            revision_dir=Path("/tmp/03_revision"),
            shared_dir=Path("/tmp/00_shared"),
        )
        assert config.compile_script is None

    def test_compile_script_can_be_set(self):
        """Verify compile_script can be explicitly set."""
        config = WriterConfig(
            project_dir=Path("/tmp"),
            manuscript_dir=Path("/tmp/01_manuscript"),
            supplementary_dir=Path("/tmp/02_supplementary"),
            revision_dir=Path("/tmp/03_revision"),
            shared_dir=Path("/tmp/00_shared"),
            compile_script=Path("/tmp/compile.sh"),
        )
        assert config.compile_script == Path("/tmp/compile.sh")


class TestWriterConfigFromDirectory:
    """Tests for WriterConfig.from_directory factory method."""

    def test_from_directory_creates_config(self):
        """Verify from_directory creates proper config."""
        project_dir = Path("/tmp/test_project")
        config = WriterConfig.from_directory(project_dir)

        assert config.project_dir == project_dir

    def test_from_directory_sets_manuscript_dir(self):
        """Verify manuscript_dir is set to 01_manuscript subdirectory."""
        project_dir = Path("/tmp/test_project")
        config = WriterConfig.from_directory(project_dir)

        assert config.manuscript_dir == project_dir / "01_manuscript"

    def test_from_directory_sets_supplementary_dir(self):
        """Verify supplementary_dir is set to 02_supplementary subdirectory."""
        project_dir = Path("/tmp/test_project")
        config = WriterConfig.from_directory(project_dir)

        assert config.supplementary_dir == project_dir / "02_supplementary"

    def test_from_directory_sets_revision_dir(self):
        """Verify revision_dir is set to 03_revision subdirectory."""
        project_dir = Path("/tmp/test_project")
        config = WriterConfig.from_directory(project_dir)

        assert config.revision_dir == project_dir / "03_revision"

    def test_from_directory_sets_shared_dir(self):
        """Verify shared_dir is set to 00_shared subdirectory."""
        project_dir = Path("/tmp/test_project")
        config = WriterConfig.from_directory(project_dir)

        assert config.shared_dir == project_dir / "00_shared"

    def test_from_directory_accepts_string(self):
        """Verify from_directory converts string to Path."""
        config = WriterConfig.from_directory("/tmp/test_project")
        assert isinstance(config.project_dir, Path)


class TestWriterConfigValidation:
    """Tests for WriterConfig.validate method."""

    def test_validate_nonexistent_project_dir_raises(self, tmp_path):
        """Verify validate raises for nonexistent project directory."""
        config = WriterConfig.from_directory(tmp_path / "nonexistent")

        with pytest.raises(ValueError, match="Project directory not found"):
            config.validate()

    def test_validate_empty_project_dir_raises(self, tmp_path):
        """Verify validate raises when no document directories exist."""
        config = WriterConfig.from_directory(tmp_path)

        with pytest.raises(ValueError, match="No document directories found"):
            config.validate()

    def test_validate_with_manuscript_dir_passes(self, tmp_path):
        """Verify validate passes with manuscript directory."""
        (tmp_path / "01_manuscript").mkdir()
        config = WriterConfig.from_directory(tmp_path)

        assert config.validate() is True

    def test_validate_with_supplementary_dir_passes(self, tmp_path):
        """Verify validate passes with supplementary directory."""
        (tmp_path / "02_supplementary").mkdir()
        config = WriterConfig.from_directory(tmp_path)

        assert config.validate() is True

    def test_validate_with_revision_dir_passes(self, tmp_path):
        """Verify validate passes with revision directory."""
        (tmp_path / "03_revision").mkdir()
        config = WriterConfig.from_directory(tmp_path)

        assert config.validate() is True

    def test_validate_with_all_dirs_passes(self, tmp_path):
        """Verify validate passes with all document directories."""
        (tmp_path / "01_manuscript").mkdir()
        (tmp_path / "02_supplementary").mkdir()
        (tmp_path / "03_revision").mkdir()
        config = WriterConfig.from_directory(tmp_path)

        assert config.validate() is True


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
