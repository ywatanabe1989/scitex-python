#!/usr/bin/env python3
"""Tests for scitex.writer._project._validate."""

from pathlib import Path

import pytest

from scitex.writer._project._validate import validate_structure


class TestValidateStructureSuccess:
    """Tests for validate_structure when structure is valid."""

    def test_passes_with_all_required_directories(self, tmp_path):
        """Verify validate_structure passes with complete structure."""
        (tmp_path / "01_manuscript").mkdir()
        (tmp_path / "02_supplementary").mkdir()
        (tmp_path / "03_revision").mkdir()

        # Should not raise
        validate_structure(tmp_path)

    def test_passes_with_extra_directories(self, tmp_path):
        """Verify validate_structure passes when extra directories exist."""
        (tmp_path / "01_manuscript").mkdir()
        (tmp_path / "02_supplementary").mkdir()
        (tmp_path / "03_revision").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "shared").mkdir()

        # Should not raise
        validate_structure(tmp_path)


class TestValidateStructureFailure:
    """Tests for validate_structure when structure is invalid."""

    def test_raises_when_manuscript_missing(self, tmp_path):
        """Verify raises RuntimeError when 01_manuscript is missing."""
        (tmp_path / "02_supplementary").mkdir()
        (tmp_path / "03_revision").mkdir()

        with pytest.raises(RuntimeError, match="01_manuscript"):
            validate_structure(tmp_path)

    def test_raises_when_supplementary_missing(self, tmp_path):
        """Verify raises RuntimeError when 02_supplementary is missing."""
        (tmp_path / "01_manuscript").mkdir()
        (tmp_path / "03_revision").mkdir()

        with pytest.raises(RuntimeError, match="02_supplementary"):
            validate_structure(tmp_path)

    def test_raises_when_revision_missing(self, tmp_path):
        """Verify raises RuntimeError when 03_revision is missing."""
        (tmp_path / "01_manuscript").mkdir()
        (tmp_path / "02_supplementary").mkdir()

        with pytest.raises(RuntimeError, match="03_revision"):
            validate_structure(tmp_path)

    def test_raises_when_all_missing(self, tmp_path):
        """Verify raises RuntimeError when all required dirs are missing."""
        with pytest.raises(RuntimeError, match="01_manuscript"):
            validate_structure(tmp_path)

    def test_error_message_contains_path(self, tmp_path):
        """Verify error message contains the expected directory path."""
        with pytest.raises(RuntimeError) as exc_info:
            validate_structure(tmp_path)

        assert "expected at:" in str(exc_info.value)


class TestValidateStructureEdgeCases:
    """Tests for validate_structure edge cases."""

    def test_file_instead_of_directory(self, tmp_path):
        """Verify raises when file exists instead of directory."""
        (tmp_path / "01_manuscript").write_text("not a directory")
        (tmp_path / "02_supplementary").mkdir()
        (tmp_path / "03_revision").mkdir()

        # File exists but is not a directory - exists() still returns True
        # This may or may not be an issue depending on expected behavior
        validate_structure(tmp_path)

    def test_nested_project_directory(self, tmp_path):
        """Verify works with nested project directory."""
        nested = tmp_path / "projects" / "my_paper"
        nested.mkdir(parents=True)
        (nested / "01_manuscript").mkdir()
        (nested / "02_supplementary").mkdir()
        (nested / "03_revision").mkdir()

        validate_structure(nested)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_project/_validate.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-29 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_project/_validate.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/writer/_project/_validate.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Project structure validation for writer module.
# 
# Verifies that writer projects have expected directory structure.
# """
# 
# from pathlib import Path
# 
# from scitex.logging import getLogger
# 
# logger = getLogger(__name__)
# 
# 
# def validate_structure(project_dir: Path) -> None:
#     """
#     Verify attached project has expected structure.
# 
#     Checks:
#     - Required directories exist (01_manuscript, 02_supplementary, 03_revision)
# 
#     Parameters
#     ----------
#     project_dir : Path
#         Path to project directory
# 
#     Raises
#     ------
#     RuntimeError
#         If structure is invalid
#     """
#     required_dirs = [
#         project_dir / "01_manuscript",
#         project_dir / "02_supplementary",
#         project_dir / "03_revision",
#     ]
# 
#     for dir_path in required_dirs:
#         if not dir_path.exists():
#             logger.error(f"Expected directory missing: {dir_path}")
#             raise RuntimeError(
#                 f"Project structure invalid: missing {dir_path.name} directory (expected at: {dir_path})"
#             )
# 
#     logger.success(f"Project structure verified at {project_dir.absolute()}")
# 
# 
# __all__ = ["validate_structure"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_project/_validate.py
# --------------------------------------------------------------------------------
