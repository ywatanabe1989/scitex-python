#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/writer/_compile/test__parser.py
# ----------------------------------------

"""
Tests for compilation output parser.

Tests parse_output function for extracting errors and warnings.
"""

import pytest
pytest.importorskip("git")
from pathlib import Path
from scitex.writer._compile._parser import parse_output


class TestParseOutput:
    """Test suite for parse_output function."""

    def test_import(self):
        """Test that parse_output can be imported."""
        assert callable(parse_output)

    def test_parse_empty_output(self):
        """Test parsing empty output returns empty lists."""
        errors, warnings = parse_output("", "")
        assert errors == []
        assert warnings == []

    def test_parse_output_with_no_log_file(self):
        """Test parsing without log file."""
        stdout = "Compilation successful"
        stderr = ""
        errors, warnings = parse_output(stdout, stderr, log_file=None)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)

    def test_parse_output_with_log_file(self):
        """Test parsing with log file path."""
        stdout = "Compilation successful"
        stderr = ""
        log_file = Path("/tmp/test.log")
        errors, warnings = parse_output(stdout, stderr, log_file=log_file)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_parser.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-29 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_parser.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/writer/_compile/_parser.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Compilation output parsing.
# 
# Parses LaTeX compilation output and log files for errors and warnings.
# """
# 
# from pathlib import Path
# from typing import Tuple, List, Optional
# 
# from scitex.logging import getLogger
# from scitex.writer.utils._parse_latex_logs import parse_compilation_output
# 
# logger = getLogger(__name__)
# 
# 
# def parse_output(
#     stdout: str,
#     stderr: str,
#     log_file: Optional[Path] = None,
# ) -> Tuple[List[str], List[str]]:
#     """
#     Parse compilation output for errors and warnings.
# 
#     Parameters
#     ----------
#     stdout : str
#         Standard output from compilation
#     stderr : str
#         Standard error from compilation
#     log_file : Path, optional
#         Path to LaTeX log file
# 
#     Returns
#     -------
#     tuple
#         (errors, warnings) as lists of strings
#     """
#     error_issues, warning_issues = parse_compilation_output(
#         stdout + stderr, log_file=log_file
#     )
# 
#     # Convert LaTeXIssue objects to strings for backward compatibility
#     errors = [str(issue) for issue in error_issues]
#     warnings = [str(issue) for issue in warning_issues]
# 
#     return errors, warnings
# 
# 
# __all__ = ["parse_output"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_parser.py
# --------------------------------------------------------------------------------
