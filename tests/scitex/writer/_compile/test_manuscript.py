#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/writer/_compile/test_manuscript.py
# ----------------------------------------

"""
Tests for manuscript compilation.

Tests compile_manuscript function with various options:
- no_figs: Exclude figures for quick compilation
- ppt2tif: PowerPoint to TIF conversion
- crop_tif: TIF cropping
- quiet/verbose: Output verbosity
- force: Force recompilation
- log_callback: Live logging
- progress_callback: Progress tracking
"""

import pytest
pytest.importorskip("git")
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from scitex.writer._compile.manuscript import compile_manuscript
from scitex.writer.dataclasses import CompilationResult


class TestCompileManuscript:
    """Test suite for compile_manuscript function."""

    def test_import(self):
        """Test that compile_manuscript can be imported."""
        from scitex.writer._compile import compile_manuscript as cm
        assert callable(cm)

    def test_signature(self):
        """Test function signature has expected parameters."""
        import inspect
        sig = inspect.signature(compile_manuscript)
        params = list(sig.parameters.keys())

        assert "project_dir" in params
        assert "timeout" in params
        assert "no_figs" in params
        assert "ppt2tif" in params
        assert "crop_tif" in params
        assert "quiet" in params
        assert "verbose" in params
        assert "force" in params
        assert "log_callback" in params
        assert "progress_callback" in params

    def test_default_parameters(self):
        """Test default parameter values."""
        import inspect
        sig = inspect.signature(compile_manuscript)

        assert sig.parameters["timeout"].default == 300
        assert sig.parameters["no_figs"].default is False
        assert sig.parameters["ppt2tif"].default is False
        assert sig.parameters["crop_tif"].default is False
        assert sig.parameters["quiet"].default is False
        assert sig.parameters["verbose"].default is False
        assert sig.parameters["force"].default is False
        assert sig.parameters["log_callback"].default is None
        assert sig.parameters["progress_callback"].default is None

    @patch("scitex.writer._compile.manuscript.run_compile")
    def test_calls_run_compile_with_manuscript_type(self, mock_run_compile):
        """Test that compile_manuscript calls run_compile with 'manuscript' doc_type."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_manuscript(project_dir)

        mock_run_compile.assert_called_once()
        args, kwargs = mock_run_compile.call_args
        assert args[0] == "manuscript"
        assert args[1] == project_dir

    @patch("scitex.writer._compile.manuscript.run_compile")
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
        compile_manuscript(project_dir, no_figs=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["no_figs"] is True

    @patch("scitex.writer._compile.manuscript.run_compile")
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
        compile_manuscript(project_dir, ppt2tif=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["ppt2tif"] is True

    @patch("scitex.writer._compile.manuscript.run_compile")
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
        compile_manuscript(project_dir, crop_tif=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["crop_tif"] is True

    @patch("scitex.writer._compile.manuscript.run_compile")
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
        compile_manuscript(project_dir, quiet=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["quiet"] is True

    @patch("scitex.writer._compile.manuscript.run_compile")
    def test_passes_verbose_option(self, mock_run_compile):
        """Test that verbose option is passed to run_compile."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_manuscript(project_dir, verbose=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["verbose"] is True

    @patch("scitex.writer._compile.manuscript.run_compile")
    def test_passes_force_option(self, mock_run_compile):
        """Test that force option is passed to run_compile."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_manuscript(project_dir, force=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["force"] is True

    @patch("scitex.writer._compile.manuscript.run_compile")
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
        compile_manuscript(
            project_dir,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        _, kwargs = mock_run_compile.call_args
        assert kwargs["log_callback"] is log_callback
        assert kwargs["progress_callback"] is progress_callback

    @patch("scitex.writer._compile.manuscript.run_compile")
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
        compile_manuscript(
            project_dir,
            no_figs=True,
            ppt2tif=True,
            crop_tif=True,
            verbose=True,
            force=True,
        )

        _, kwargs = mock_run_compile.call_args
        assert kwargs["no_figs"] is True
        assert kwargs["ppt2tif"] is True
        assert kwargs["crop_tif"] is True
        assert kwargs["verbose"] is True
        assert kwargs["force"] is True

    @patch("scitex.writer._compile.manuscript.run_compile")
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
        result = compile_manuscript(project_dir)

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
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/manuscript.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-08 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/manuscript.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/writer/_compile/manuscript.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Manuscript compilation function.
# 
# Provides manuscript-specific compilation with options for:
# - Figure exclusion for quick compilation
# - PowerPoint to TIF conversion
# - TIF cropping
# - Verbose/quiet modes
# - Force recompilation
# """
# 
# from pathlib import Path
# from typing import Optional, Callable
# from ._runner import run_compile
# from ..dataclasses import CompilationResult
# 
# 
# def compile_manuscript(
#     project_dir: Path,
#     timeout: int = 300,
#     no_figs: bool = False,
#     ppt2tif: bool = False,
#     crop_tif: bool = False,
#     quiet: bool = False,
#     verbose: bool = False,
#     force: bool = False,
#     log_callback: Optional[Callable[[str], None]] = None,
#     progress_callback: Optional[Callable[[int, str], None]] = None,
# ) -> CompilationResult:
#     """
#     Compile manuscript document with optional callbacks.
# 
#     Parameters
#     ----------
#     project_dir : Path
#         Path to writer project directory
#     timeout : int
#         Timeout in seconds
#     no_figs : bool
#         Exclude figures for quick compilation
#     ppt2tif : bool
#         Convert PowerPoint to TIF on WSL
#     crop_tif : bool
#         Crop TIF images to remove excess whitespace
#     quiet : bool
#         Suppress detailed logs for LaTeX compilation
#     verbose : bool
#         Show detailed logs for LaTeX compilation
#     force : bool
#         Force full recompilation, ignore cache
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
#     >>> from scitex.writer._compile import compile_manuscript
#     >>>
#     >>> # Quick compilation without figures
#     >>> result = compile_manuscript(
#     ...     project_dir=Path("~/my-paper"),
#     ...     no_figs=True,
#     ...     quiet=True
#     ... )
#     >>>
#     >>> # Full compilation with PowerPoint conversion
#     >>> result = compile_manuscript(
#     ...     project_dir=Path("~/my-paper"),
#     ...     ppt2tif=True,
#     ...     crop_tif=True,
#     ...     verbose=True
#     ... )
#     """
#     return run_compile(
#         "manuscript",
#         project_dir,
#         timeout=timeout,
#         no_figs=no_figs,
#         ppt2tif=ppt2tif,
#         crop_tif=crop_tif,
#         quiet=quiet,
#         verbose=verbose,
#         force=force,
#         log_callback=log_callback,
#         progress_callback=progress_callback,
#     )
# 
# 
# __all__ = ["compile_manuscript"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/manuscript.py
# --------------------------------------------------------------------------------
