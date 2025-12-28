#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/writer/_compile/test_revision.py
# ----------------------------------------

"""
Tests for revision response compilation.

Tests compile_revision function with various options:
- track_changes: Enable change tracking (diff highlighting)
- log_callback: Live logging
- progress_callback: Progress tracking
"""

import pytest
pytest.importorskip("git")
from pathlib import Path
from unittest.mock import Mock, patch
from scitex.writer._compile.revision import compile_revision
from scitex.writer.dataclasses import CompilationResult


class TestCompileRevision:
    """Test suite for compile_revision function."""

    def test_import(self):
        """Test that compile_revision can be imported."""
        from scitex.writer._compile import compile_revision as cr
        assert callable(cr)

    def test_signature(self):
        """Test function signature has expected parameters."""
        import inspect
        sig = inspect.signature(compile_revision)
        params = list(sig.parameters.keys())

        assert "project_dir" in params
        assert "track_changes" in params
        assert "timeout" in params
        assert "log_callback" in params
        assert "progress_callback" in params

    def test_default_parameters(self):
        """Test default parameter values."""
        import inspect
        sig = inspect.signature(compile_revision)

        assert sig.parameters["track_changes"].default is False
        assert sig.parameters["timeout"].default == 300
        assert sig.parameters["log_callback"].default is None
        assert sig.parameters["progress_callback"].default is None

    @patch("scitex.writer._compile.revision.run_compile")
    def test_calls_run_compile_with_revision_type(self, mock_run_compile):
        """Test that compile_revision calls run_compile with 'revision' doc_type."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_revision(project_dir)

        mock_run_compile.assert_called_once()
        args, kwargs = mock_run_compile.call_args
        assert args[0] == "revision"
        assert args[1] == project_dir

    @patch("scitex.writer._compile.revision.run_compile")
    def test_passes_track_changes_option(self, mock_run_compile):
        """Test that track_changes option is passed to run_compile."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_revision(project_dir, track_changes=True)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["track_changes"] is True

    @patch("scitex.writer._compile.revision.run_compile")
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
        compile_revision(
            project_dir,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        _, kwargs = mock_run_compile.call_args
        assert kwargs["log_callback"] is log_callback
        assert kwargs["progress_callback"] is progress_callback

    @patch("scitex.writer._compile.revision.run_compile")
    def test_passes_timeout(self, mock_run_compile):
        """Test that timeout is passed to run_compile."""
        mock_run_compile.return_value = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
        )

        project_dir = Path("/tmp/test-project")
        compile_revision(project_dir, timeout=600)

        _, kwargs = mock_run_compile.call_args
        assert kwargs["timeout"] == 600

    @patch("scitex.writer._compile.revision.run_compile")
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
        result = compile_revision(project_dir)

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
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/revision.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-08 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/revision.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/writer/_compile/revision.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Revision response compilation function.
# 
# Provides revision-specific compilation with options for:
# - Change tracking (diff highlighting)
# """
# 
# from pathlib import Path
# from typing import Optional, Callable
# from ._runner import run_compile
# from ..dataclasses import CompilationResult
# 
# 
# def compile_revision(
#     project_dir: Path,
#     track_changes: bool = False,
#     timeout: int = 300,
#     log_callback: Optional[Callable[[str], None]] = None,
#     progress_callback: Optional[Callable[[int, str], None]] = None,
# ) -> CompilationResult:
#     """
#     Compile revision responses with optional callbacks.
# 
#     Parameters
#     ----------
#     project_dir : Path
#         Path to writer project directory
#     track_changes : bool
#         Whether to enable change tracking (diff highlighting)
#     timeout : int
#         Timeout in seconds
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
#     >>> from scitex.writer._compile import compile_revision
#     >>>
#     >>> # Standard revision compilation
#     >>> result = compile_revision(
#     ...     project_dir=Path("~/my-paper")
#     ... )
#     >>>
#     >>> # Compilation with change tracking
#     >>> result = compile_revision(
#     ...     project_dir=Path("~/my-paper"),
#     ...     track_changes=True
#     ... )
#     """
#     return run_compile(
#         "revision",
#         project_dir,
#         timeout=timeout,
#         track_changes=track_changes,
#         log_callback=log_callback,
#         progress_callback=progress_callback,
#     )
# 
# 
# __all__ = ["compile_revision"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/revision.py
# --------------------------------------------------------------------------------
