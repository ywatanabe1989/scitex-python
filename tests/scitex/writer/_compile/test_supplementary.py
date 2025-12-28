#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/writer/_compile/test_supplementary.py
# ----------------------------------------

"""
Tests for supplementary materials compilation.

Tests compile_supplementary function with various options:
- no_figs: Exclude figures (default includes figures)
- ppt2tif: PowerPoint to TIF conversion
- crop_tif: TIF cropping
- quiet: Output verbosity
- log_callback: Live logging
- progress_callback: Progress tracking
"""

import pytest
pytest.importorskip("git")
from pathlib import Path
from unittest.mock import Mock, patch
from scitex.writer._compile.supplementary import compile_supplementary
from scitex.writer.dataclasses import CompilationResult


class TestCompileSupplementary:
    """Test suite for compile_supplementary function."""

    def test_import(self):
        """Test that compile_supplementary can be imported."""
        from scitex.writer._compile import compile_supplementary as cs
        assert callable(cs)

    def test_signature(self):
        """Test function signature has expected parameters."""
        import inspect
        sig = inspect.signature(compile_supplementary)
        params = list(sig.parameters.keys())

        assert "project_dir" in params
        assert "timeout" in params
        assert "no_figs" in params
        assert "ppt2tif" in params
        assert "crop_tif" in params
        assert "quiet" in params
        assert "log_callback" in params
        assert "progress_callback" in params

    def test_default_parameters(self):
        """Test default parameter values."""
        import inspect
        sig = inspect.signature(compile_supplementary)

        assert sig.parameters["timeout"].default == 300
        assert sig.parameters["no_figs"].default is False
        assert sig.parameters["ppt2tif"].default is False
        assert sig.parameters["crop_tif"].default is False
        assert sig.parameters["quiet"].default is False
        assert sig.parameters["log_callback"].default is None
        assert sig.parameters["progress_callback"].default is None

    @patch("scitex.writer._compile.supplementary.run_compile")
    def test_calls_run_compile_with_supplementary_type(self, mock_run_compile):
        """Test that compile_supplementary calls run_compile with 'supplementary' doc_type."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_supplementary(project_dir)

        mock_run_compile.assert_called_once()
        args, kwargs = mock_run_compile.call_args
        assert args[0] == "supplementary"
        assert args[1] == project_dir

    @patch("scitex.writer._compile.supplementary.run_compile")
    def test_passes_no_figs_option(self, mock_run_compile):
        """Test that no_figs option is passed to run_compile."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_supplementary(project_dir, no_figs=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["no_figs"] is True

    @patch("scitex.writer._compile.supplementary.run_compile")
    def test_passes_ppt2tif_option(self, mock_run_compile):
        """Test that ppt2tif option is passed to run_compile."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_supplementary(project_dir, ppt2tif=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["ppt2tif"] is True

    @patch("scitex.writer._compile.supplementary.run_compile")
    def test_passes_crop_tif_option(self, mock_run_compile):
        """Test that crop_tif option is passed to run_compile."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_supplementary(project_dir, crop_tif=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["crop_tif"] is True

    @patch("scitex.writer._compile.supplementary.run_compile")
    def test_passes_quiet_option(self, mock_run_compile):
        """Test that quiet option is passed to run_compile."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_supplementary(project_dir, quiet=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["quiet"] is True

    @patch("scitex.writer._compile.supplementary.run_compile")
    def test_passes_callbacks(self, mock_run_compile):
        """Test that callbacks are passed to run_compile."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        log_callback = Mock()
        progress_callback = Mock()

        project_dir = Path("/tmp/test-project")
        compile_supplementary(
            project_dir,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        _, kwargs = mock_run_compile.call_args
        assert kwargs["log_callback"] is log_callback
        assert kwargs["progress_callback"] is progress_callback

    @patch("scitex.writer._compile.supplementary.run_compile")
    def test_passes_multiple_options(self, mock_run_compile):
        """Test that multiple options are passed correctly."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_supplementary(
            project_dir,
            ppt2tif=True,
            crop_tif=True,
            quiet=True,
        )

        _, kwargs = mock_run_compile.call_args
        assert kwargs["ppt2tif"] is True
        assert kwargs["crop_tif"] is True
        assert kwargs["quiet"] is True

    @patch("scitex.writer._compile.supplementary.run_compile")
    def test_returns_compilation_result(self, mock_run_compile):
        """Test that function returns CompilationResult."""
        expected_result = CompilationResult(
            success=True,
            exit_code=0,
            stdout="Test output",
            stderr="",
            duration=2.5,
        )
        mock_run_compile.return_value = expected_result

        project_dir = Path("/tmp/test-project")
        result = compile_supplementary(project_dir)

        assert result is expected_result
        assert result.success is True
        assert result.exit_code == 0
        assert result.duration == 2.5


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/supplementary.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-08 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/supplementary.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/writer/_compile/supplementary.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Supplementary materials compilation function.
# 
# Provides supplementary-specific compilation with options for:
# - Figure inclusion (default)
# - PowerPoint to TIF conversion
# - TIF cropping
# - Quiet mode
# """
# 
# from pathlib import Path
# from typing import Optional, Callable
# from ._runner import run_compile
# from ..dataclasses import CompilationResult
# 
# 
# def compile_supplementary(
#     project_dir: Path,
#     timeout: int = 300,
#     no_figs: bool = False,
#     ppt2tif: bool = False,
#     crop_tif: bool = False,
#     quiet: bool = False,
#     log_callback: Optional[Callable[[str], None]] = None,
#     progress_callback: Optional[Callable[[int, str], None]] = None,
# ) -> CompilationResult:
#     """
#     Compile supplementary materials with optional callbacks.
# 
#     Parameters
#     ----------
#     project_dir : Path
#         Path to writer project directory
#     timeout : int
#         Timeout in seconds
#     no_figs : bool
#         Exclude figures (default includes figures)
#     ppt2tif : bool
#         Convert PowerPoint to TIF on WSL
#     crop_tif : bool
#         Crop TIF images to remove excess whitespace
#     quiet : bool
#         Suppress detailed logs for LaTeX compilation
#     log_callback : Optional[Callable[[str], None]]
#         Called with each log line
#     progress_callback : Optional[Callable[[int, str], None]]
#         Called with progress updates (percent, step)
# 
#     Returns
#     -------
#     CompilationResult
#         Compilation status and outputs
# 
#     Examples
#     --------
#     >>> from pathlib import Path
#     >>> from scitex.writer._compile import compile_supplementary
#     >>>
#     >>> # Standard compilation with figures
#     >>> result = compile_supplementary(
#     ...     project_dir=Path("~/my-paper"),
#     ...     ppt2tif=True,
#     ...     quiet=False
#     ... )
#     >>>
#     >>> # Quick compilation without figures
#     >>> result = compile_supplementary(
#     ...     project_dir=Path("~/my-paper"),
#     ...     no_figs=True,
#     ...     quiet=True
#     ... )
#     """
#     return run_compile(
#         "supplementary",
#         project_dir,
#         timeout=timeout,
#         no_figs=no_figs,
#         ppt2tif=ppt2tif,
#         crop_tif=crop_tif,
#         quiet=quiet,
#         log_callback=log_callback,
#         progress_callback=progress_callback,
#     )
# 
# 
# __all__ = ["compile_supplementary"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/supplementary.py
# --------------------------------------------------------------------------------
