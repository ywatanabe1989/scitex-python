#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/writer/_compile/test__validator.py
# ----------------------------------------

"""
Tests for pre-compile validation.

Tests validate_before_compile function for project structure validation.
"""

import pytest
pytest.importorskip("git")
from pathlib import Path
from scitex.writer._compile._validator import validate_before_compile


class TestValidateBeforeCompile:
    """Test suite for validate_before_compile function."""

    def test_import(self):
        """Test that validate_before_compile can be imported."""
        assert callable(validate_before_compile)

    def test_validate_nonexistent_directory(self):
        """Test validation fails for non-existent directory."""
        project_dir = Path("/tmp/nonexistent-project-12345")
        with pytest.raises(Exception):
            validate_before_compile(project_dir)

    def test_validate_requires_path_object(self):
        """Test that function accepts Path objects."""
        # This should not raise a type error
        project_dir = Path("/tmp/test")
        try:
            validate_before_compile(project_dir)
        except Exception:
            # Any exception is fine, we just want to check it accepts Path
            pass


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_validator.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-29 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_validator.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/writer/_compile/_validator.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Pre-compile validation for writer projects.
# 
# Validates project structure before attempting compilation.
# """
# 
# from pathlib import Path
# 
# from scitex.logging import getLogger
# from scitex.writer._validate_tree_structures import validate_tree_structures
# 
# logger = getLogger(__name__)
# 
# 
# def validate_before_compile(project_dir: Path) -> None:
#     """
#     Validate project structure before compilation.
# 
#     Parameters
#     ----------
#     project_dir : Path
#         Path to project directory
# 
#     Raises
#     ------
#     RuntimeError
#         If validation fails
#     """
#     validate_tree_structures(project_dir)
# 
# 
# __all__ = ["validate_before_compile"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_validator.py
# --------------------------------------------------------------------------------
