#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/writer/_compile/test__runner.py
# ----------------------------------------

"""
Tests for compilation script runner.

Tests run_compile function with various options and document types.
"""

import pytest

pytest.importorskip("git")
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from scitex.writer._compile._runner import run_compile, _get_compile_script
from scitex.writer.dataclasses import CompilationResult


class TestGetCompileScript:
    """Test suite for _get_compile_script helper function."""

    def test_manuscript_script_path(self):
        """Test manuscript script path generation."""
        project_dir = Path("/tmp/test-project")
        script = _get_compile_script(project_dir, "manuscript")
        assert script == project_dir / "scripts" / "shell" / "compile_manuscript.sh"

    def test_supplementary_script_path(self):
        """Test supplementary script path generation."""
        project_dir = Path("/tmp/test-project")
        script = _get_compile_script(project_dir, "supplementary")
        assert script == project_dir / "scripts" / "shell" / "compile_supplementary.sh"

    def test_revision_script_path(self):
        """Test revision script path generation."""
        project_dir = Path("/tmp/test-project")
        script = _get_compile_script(project_dir, "revision")
        assert script == project_dir / "scripts" / "shell" / "compile_revision.sh"


class TestRunCompile:
    """Test suite for run_compile function."""

    def test_signature(self):
        """Test function signature has expected parameters."""
        import inspect

        sig = inspect.signature(run_compile)
        params = list(sig.parameters.keys())

        assert "doc_type" in params
        assert "project_dir" in params
        assert "timeout" in params
        assert "track_changes" in params
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

        sig = inspect.signature(run_compile)

        assert sig.parameters["timeout"].default == 300
        assert sig.parameters["track_changes"].default is False
        assert sig.parameters["no_figs"].default is False
        assert sig.parameters["ppt2tif"].default is False
        assert sig.parameters["crop_tif"].default is False
        assert sig.parameters["quiet"].default is False
        assert sig.parameters["verbose"].default is False
        assert sig.parameters["force"].default is False
        assert sig.parameters["log_callback"].default is None
        assert sig.parameters["progress_callback"].default is None

    @patch("scitex.writer._compile._runner.validate_before_compile")
    @patch("scitex.writer._compile._runner._get_compile_script")
    @patch("scitex.writer._compile._runner.sh")
    @patch("scitex.writer._compile._runner._find_output_files")
    @patch("scitex.writer._compile._runner.parse_output")
    def test_manuscript_with_no_figs_option(
        self,
        mock_parse,
        mock_find_files,
        mock_sh,
        mock_get_script,
        mock_validate,
    ):
        """Test manuscript compilation with no_figs option."""
        project_dir = Path("/tmp/test-project")
        script_path = project_dir / "scripts" / "shell" / "compile_manuscript.sh"

        mock_get_script.return_value = script_path
        mock_find_files.return_value = (None, None, None)
        mock_parse.return_value = ([], [])
        mock_sh.return_value = {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }

        with patch("pathlib.Path.exists", return_value=True):
            with patch("os.chdir"):
                with patch("pathlib.Path.cwd", return_value=project_dir):
                    result = run_compile(
                        "manuscript",
                        project_dir,
                        no_figs=True,
                    )

        # Verify sh was called with correct command
        mock_sh.assert_called_once()
        call_args = mock_sh.call_args[0][0]
        assert str(script_path) in call_args
        assert "--no_figs" in call_args

    @patch("scitex.writer._compile._runner.validate_before_compile")
    @patch("scitex.writer._compile._runner._get_compile_script")
    @patch("scitex.writer._compile._runner.sh")
    @patch("scitex.writer._compile._runner._find_output_files")
    @patch("scitex.writer._compile._runner.parse_output")
    def test_manuscript_with_multiple_options(
        self,
        mock_parse,
        mock_find_files,
        mock_sh,
        mock_get_script,
        mock_validate,
    ):
        """Test manuscript compilation with multiple options."""
        project_dir = Path("/tmp/test-project")
        script_path = project_dir / "scripts" / "shell" / "compile_manuscript.sh"

        mock_get_script.return_value = script_path
        mock_find_files.return_value = (None, None, None)
        mock_parse.return_value = ([], [])
        mock_sh.return_value = {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }

        with patch("pathlib.Path.exists", return_value=True):
            with patch("os.chdir"):
                with patch("pathlib.Path.cwd", return_value=project_dir):
                    result = run_compile(
                        "manuscript",
                        project_dir,
                        no_figs=True,
                        ppt2tif=True,
                        crop_tif=True,
                        verbose=True,
                        force=True,
                    )

        # Verify sh was called with all options
        call_args = mock_sh.call_args[0][0]
        assert "--no_figs" in call_args
        assert "--ppt2tif" in call_args
        assert "--crop_tif" in call_args
        assert "--verbose" in call_args
        assert "--force" in call_args

    @patch("scitex.writer._compile._runner.validate_before_compile")
    @patch("scitex.writer._compile._runner._get_compile_script")
    @patch("scitex.writer._compile._runner.sh")
    @patch("scitex.writer._compile._runner._find_output_files")
    @patch("scitex.writer._compile._runner.parse_output")
    def test_supplementary_with_figs_option(
        self,
        mock_parse,
        mock_find_files,
        mock_sh,
        mock_get_script,
        mock_validate,
    ):
        """Test supplementary compilation with figs option."""
        project_dir = Path("/tmp/test-project")
        script_path = project_dir / "scripts" / "shell" / "compile_supplementary.sh"

        mock_get_script.return_value = script_path
        mock_find_files.return_value = (None, None, None)
        mock_parse.return_value = ([], [])
        mock_sh.return_value = {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }

        with patch("pathlib.Path.exists", return_value=True):
            with patch("os.chdir"):
                with patch("pathlib.Path.cwd", return_value=project_dir):
                    result = run_compile(
                        "supplementary",
                        project_dir,
                        no_figs=False,  # Include figures
                    )

        # Verify sh was called with --figs option
        call_args = mock_sh.call_args[0][0]
        assert "--figs" in call_args

    @patch("scitex.writer._compile._runner.validate_before_compile")
    @patch("scitex.writer._compile._runner._get_compile_script")
    @patch("scitex.writer._compile._runner.sh")
    @patch("scitex.writer._compile._runner._find_output_files")
    @patch("scitex.writer._compile._runner.parse_output")
    def test_revision_with_track_changes(
        self,
        mock_parse,
        mock_find_files,
        mock_sh,
        mock_get_script,
        mock_validate,
    ):
        """Test revision compilation with track_changes option."""
        project_dir = Path("/tmp/test-project")
        script_path = project_dir / "scripts" / "shell" / "compile_revision.sh"

        mock_get_script.return_value = script_path
        mock_find_files.return_value = (None, None, None)
        mock_parse.return_value = ([], [])
        mock_sh.return_value = {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }

        with patch("pathlib.Path.exists", return_value=True):
            with patch("os.chdir"):
                with patch("pathlib.Path.cwd", return_value=project_dir):
                    result = run_compile(
                        "revision",
                        project_dir,
                        track_changes=True,
                    )

        # Verify sh was called with --track-changes option
        call_args = mock_sh.call_args[0][0]
        assert "--track-changes" in call_args


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_runner.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-29 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_runner.py
# # ----------------------------------------
# from __future__ import annotations
# import os
#
# __FILE__ = "./src/scitex/writer/_compile/_runner.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# """
# Compilation script execution.
#
# Executes LaTeX compilation scripts and captures results.
# """
#
# from pathlib import Path
# from datetime import datetime
# from typing import Optional, Callable
# import subprocess
# import time
# import fcntl
#
# from scitex.logging import getLogger
# from scitex.sh import sh
# from scitex.writer.dataclasses.config import DOC_TYPE_DIRS
# from scitex.writer.dataclasses import CompilationResult
# from ._validator import validate_before_compile
# from ._parser import parse_output
#
# logger = getLogger(__name__)
#
#
# def _get_compile_script(project_dir: Path, doc_type: str) -> Path:
#     """
#     Get compile script path for document type.
#
#     Parameters
#     ----------
#     project_dir : Path
#         Path to project directory
#     doc_type : str
#         Document type ('manuscript', 'supplementary', 'revision')
#
#     Returns
#     -------
#     Path
#         Path to compilation script
#     """
#     script_map = {
#         "manuscript": project_dir / "scripts" / "shell" / "compile_manuscript.sh",
#         "supplementary": project_dir / "scripts" / "shell" / "compile_supplementary.sh",
#         "revision": project_dir / "scripts" / "shell" / "compile_revision.sh",
#     }
#     return script_map.get(doc_type)
#
#
# def _find_output_files(
#     project_dir: Path,
#     doc_type: str,
# ) -> tuple:
#     """
#     Find generated output files after compilation.
#
#     Parameters
#     ----------
#     project_dir : Path
#         Path to project directory
#     doc_type : str
#         Document type
#
#     Returns
#     -------
#     tuple
#         (output_pdf, diff_pdf, log_file)
#     """
#     doc_dir = project_dir / DOC_TYPE_DIRS[doc_type]
#
#     # Find generated PDF
#     pdf_name = f"{doc_type}.pdf"
#     potential_pdf = doc_dir / pdf_name
#     output_pdf = potential_pdf if potential_pdf.exists() else None
#
#     # Check for diff PDF
#     diff_name = f"{doc_type}_diff.pdf"
#     potential_diff = doc_dir / diff_name
#     diff_pdf = potential_diff if potential_diff.exists() else None
#
#     # Find log file
#     log_dir = doc_dir / "logs"
#     log_file = None
#     if log_dir.exists():
#         log_files = list(log_dir.glob("*.log"))
#         if log_files:
#             log_file = max(log_files, key=lambda p: p.stat().st_mtime)
#
#     return output_pdf, diff_pdf, log_file
#
#
# def _execute_with_callbacks(
#     command: list,
#     cwd: Path,
#     timeout: int,
#     log_callback: Optional[Callable[[str], None]] = None,
# ) -> dict:
#     """
#     Execute command with line-by-line output capture and callbacks.
#
#     Parameters
#     ----------
#     command : list
#         Command to execute as list
#     cwd : Path
#         Working directory
#     timeout : int
#         Timeout in seconds
#     log_callback : Optional[Callable[[str], None]]
#         Called with each output line
#
#     Returns
#     -------
#     dict
#         Dict with stdout, stderr, exit_code, success
#     """
#     # Set environment for unbuffered output
#     env = os.environ.copy()
#     env["PYTHONUNBUFFERED"] = "1"
#
#     process = subprocess.Popen(
#         command,
#         shell=False,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         bufsize=0,  # Unbuffered
#         cwd=str(cwd),
#         env=env,
#     )
#
#     stdout_lines = []
#     stderr_lines = []
#     start_time = time.time()
#
#     # Make file descriptors non-blocking
#     def make_non_blocking(fd):
#         flags = fcntl.fcntl(fd, fcntl.F_GETFL)
#         fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
#
#     make_non_blocking(process.stdout)
#     make_non_blocking(process.stderr)
#
#     stdout_buffer = b""
#     stderr_buffer = b""
#
#     try:
#         while True:
#             # Check timeout
#             if timeout and (time.time() - start_time) > timeout:
#                 process.kill()
#                 timeout_msg = f"[ERROR] Command timed out after {timeout} seconds"
#                 if log_callback:
#                     log_callback(timeout_msg)
#                 stderr_lines.append(timeout_msg)
#                 break
#
#             # Check if process has finished
#             poll_result = process.poll()
#
#             # Read from stdout
#             try:
#                 chunk = process.stdout.read()
#                 if chunk:
#                     stdout_buffer += chunk
#                     # Process complete lines
#                     while b"\n" in stdout_buffer:
#                         line, stdout_buffer = stdout_buffer.split(b"\n", 1)
#                         line_str = line.decode("utf-8", errors="replace")
#                         stdout_lines.append(line_str)
#                         if log_callback:
#                             log_callback(line_str)
#             except (IOError, BlockingIOError):
#                 pass
#
#             # Read from stderr
#             try:
#                 chunk = process.stderr.read()
#                 if chunk:
#                     stderr_buffer += chunk
#                     # Process complete lines
#                     while b"\n" in stderr_buffer:
#                         line, stderr_buffer = stderr_buffer.split(b"\n", 1)
#                         line_str = line.decode("utf-8", errors="replace")
#                         stderr_lines.append(line_str)
#                         if log_callback:
#                             log_callback(f"[STDERR] {line_str}")
#             except (IOError, BlockingIOError):
#                 pass
#
#             # If process finished, do final read and break
#             if poll_result is not None:
#                 # Process remaining buffer content
#                 if stdout_buffer:
#                     line_str = stdout_buffer.decode("utf-8", errors="replace")
#                     stdout_lines.append(line_str)
#                     if log_callback:
#                         log_callback(line_str)
#
#                 if stderr_buffer:
#                     line_str = stderr_buffer.decode("utf-8", errors="replace")
#                     stderr_lines.append(line_str)
#                     if log_callback:
#                         log_callback(f"[STDERR] {line_str}")
#
#                 break
#
#             # Small sleep to prevent CPU spinning
#             time.sleep(0.05)
#
#     except Exception as e:
#         process.kill()
#         raise
#
#     return {
#         "stdout": "\n".join(stdout_lines),
#         "stderr": "\n".join(stderr_lines),
#         "exit_code": process.returncode,
#         "success": process.returncode == 0,
#     }
#
#
# def run_compile(
#     doc_type: str,
#     project_dir: Path,
#     timeout: int = 300,
#     track_changes: bool = False,
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
#     Run compilation script and parse results with optional callbacks.
#
#     Parameters
#     ----------
#     doc_type : str
#         Document type ('manuscript', 'supplementary', 'revision')
#     project_dir : Path
#         Path to project directory (containing 01_manuscript/, etc.)
#     timeout : int
#         Timeout in seconds
#     track_changes : bool
#         Enable change tracking (revision only)
#     no_figs : bool
#         Exclude figures for quick compilation (manuscript only)
#     ppt2tif : bool
#         Convert PowerPoint to TIF on WSL
#     crop_tif : bool
#         Crop TIF images to remove excess whitespace
#     quiet : bool
#         Suppress detailed logs for LaTeX compilation
#     verbose : bool
#         Show detailed logs for LaTeX compilation
#     force : bool
#         Force full recompilation, ignore cache (manuscript only)
#     log_callback : Optional[Callable[[str], None]]
#         Called with each log line
#     progress_callback : Optional[Callable[[int, str], None]]
#         Called with progress updates (percent, step)
#
#     Returns
#     -------
#     CompilationResult
#         Compilation status and outputs
#     """
#     start_time = datetime.now()
#     project_dir = Path(project_dir).absolute()
#
#     # Helper for progress tracking
#     def progress(percent: int, step: str):
#         if progress_callback:
#             progress_callback(percent, step)
#         logger.info(f"Progress: {percent}% - {step}")
#
#     # Helper for logging
#     def log(message: str):
#         if log_callback:
#             log_callback(message)
#         logger.info(message)
#
#     # Progress: Starting
#     progress(0, "Starting compilation...")
#     log("[INFO] Starting LaTeX compilation...")
#
#     # Validate project structure before compilation
#     try:
#         progress(5, "Validating project structure...")
#         validate_before_compile(project_dir)
#         log("[INFO] Project structure validated")
#     except Exception as e:
#         error_msg = f"[ERROR] Validation failed: {e}"
#         log(error_msg)
#         return CompilationResult(
#             success=False,
#             exit_code=1,
#             stdout="",
#             stderr=str(e),
#             duration=0.0,
#         )
#
#     # Get compile script
#     compile_script = _get_compile_script(project_dir, doc_type)
#     if not compile_script or not compile_script.exists():
#         error_msg = f"[ERROR] Compilation script not found: {compile_script}"
#         log(error_msg)
#         return CompilationResult(
#             success=False,
#             exit_code=127,
#             stdout="",
#             stderr=error_msg,
#             duration=0.0,
#         )
#
#     # Build command
#     progress(10, "Preparing compilation command...")
#     script_path = compile_script.absolute()
#     cmd = [str(script_path)]
#
#     # Add document-specific options
#     if doc_type == "revision":
#         if track_changes:
#             cmd.append("--track-changes")
#
#     elif doc_type == "manuscript":
#         if no_figs:
#             cmd.append("--no_figs")
#         if ppt2tif:
#             cmd.append("--ppt2tif")
#         if crop_tif:
#             cmd.append("--crop_tif")
#         if quiet:
#             cmd.append("--quiet")
#         elif verbose:
#             cmd.append("--verbose")
#         if force:
#             cmd.append("--force")
#
#     elif doc_type == "supplementary":
#         if not no_figs:  # For supplementary, --figs means include figures (default)
#             cmd.append("--figs")
#         if ppt2tif:
#             cmd.append("--ppt2tif")
#         if crop_tif:
#             cmd.append("--crop_tif")
#         if quiet:
#             cmd.append("--quiet")
#
#     log(f"[INFO] Running: {' '.join(cmd)}")
#     log(f"[INFO] Working directory: {project_dir}")
#
#     try:
#         cwd_original = Path.cwd()
#         os.chdir(project_dir)
#
#         try:
#             progress(15, "Executing LaTeX compilation...")
#
#             # Use callbacks version if callbacks provided
#             if log_callback:
#                 result_dict = _execute_with_callbacks(
#                     command=cmd,
#                     cwd=project_dir,
#                     timeout=timeout,
#                     log_callback=log_callback,
#                 )
#             else:
#                 # Fallback to original sh() implementation
#                 result_dict = sh(
#                     cmd,
#                     verbose=True,
#                     return_as="dict",
#                     timeout=timeout,
#                     stream_output=True,
#                 )
#
#             result = type(
#                 "Result",
#                 (),
#                 {
#                     "returncode": result_dict["exit_code"],
#                     "stdout": result_dict["stdout"],
#                     "stderr": result_dict["stderr"],
#                 },
#             )()
#
#             duration = (datetime.now() - start_time).total_seconds()
#         finally:
#             os.chdir(cwd_original)
#
#         # Find output files
#         if result.returncode == 0:
#             progress(90, "Compilation successful, locating output files...")
#             log("[INFO] Compilation succeeded, checking output files...")
#             output_pdf, diff_pdf, log_file = _find_output_files(project_dir, doc_type)
#             if output_pdf:
#                 log(f"[SUCCESS] PDF generated: {output_pdf}")
#         else:
#             output_pdf, diff_pdf, log_file = None, None, None
#             log(f"[ERROR] Compilation failed with exit code {result.returncode}")
#
#         # Parse errors and warnings
#         progress(95, "Parsing compilation logs...")
#         errors, warnings = parse_output(result.stdout, result.stderr, log_file=log_file)
#
#         compilation_result = CompilationResult(
#             success=(result.returncode == 0),
#             exit_code=result.returncode,
#             stdout=result.stdout,
#             stderr=result.stderr,
#             output_pdf=output_pdf,
#             diff_pdf=diff_pdf,
#             log_file=log_file,
#             duration=duration,
#             errors=errors,
#             warnings=warnings,
#         )
#
#         if compilation_result.success:
#             progress(100, "Complete!")
#             log(f"[SUCCESS] Compilation succeeded in {duration:.2f}s")
#         else:
#             progress(100, "Compilation failed")
#             if errors:
#                 log(f"[ERROR] Found {len(errors)} errors")
#
#         return compilation_result
#
#     except Exception as e:
#         duration = (datetime.now() - start_time).total_seconds()
#         logger.error(f"Compilation error: {e}")
#         return CompilationResult(
#             success=False,
#             exit_code=1,
#             stdout="",
#             stderr=str(e),
#             duration=duration,
#         )
#
#
# __all__ = ["run_compile"]
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_runner.py
# --------------------------------------------------------------------------------
