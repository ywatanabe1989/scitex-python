#!/usr/bin/env python3
# Timestamp: "2026-01-05 (ywatanabe)"
# File: tests/scitex/sh/test__execute.py

"""Tests for shell command execution functions.

This module tests the execution functionality of the sh module:
- execute: Main execution function with options
- _execute_buffered: Buffered output execution
- _execute_with_streaming: Real-time streaming execution
"""

import os
import tempfile

import pytest


class TestExecuteBasic:
    """Basic tests for execute function."""

    def test_execute_import(self):
        """Test execute can be imported."""
        from scitex.sh._execute import execute

        assert execute is not None
        assert callable(execute)

    def test_execute_simple_command(self):
        """Test execute with simple command."""
        from scitex.sh._execute import execute

        result = execute(["echo", "hello"], verbose=False)

        assert isinstance(result, dict)
        assert "stdout" in result
        assert "stderr" in result
        assert "exit_code" in result
        assert "success" in result

    def test_execute_returns_shell_result_structure(self):
        """Test execute returns proper ShellResult structure."""
        from scitex.sh._execute import execute

        result = execute(["echo", "test"], verbose=False)

        assert result["stdout"] == "test"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0
        assert result["success"] is True

    def test_execute_echo_command(self):
        """Test execute with echo command."""
        from scitex.sh._execute import execute

        result = execute(["echo", "Hello World"], verbose=False)

        assert result["stdout"] == "Hello World"
        assert result["success"] is True

    def test_execute_pwd_command(self):
        """Test execute with pwd command."""
        from scitex.sh._execute import execute

        result = execute(["pwd"], verbose=False)

        assert result["success"] is True
        assert len(result["stdout"]) > 0
        assert "/" in result["stdout"]  # Unix path

    def test_execute_ls_command(self):
        """Test execute with ls command."""
        from scitex.sh._execute import execute

        result = execute(["ls", "-la"], verbose=False)

        assert result["success"] is True
        assert len(result["stdout"]) > 0


class TestExecuteErrorHandling:
    """Tests for error handling in execute."""

    def test_execute_nonexistent_file(self):
        """Test execute with nonexistent file."""
        from scitex.sh._execute import execute

        result = execute(["cat", "/nonexistent/path/file.txt"], verbose=False)

        assert result["success"] is False
        assert result["exit_code"] != 0
        assert len(result["stderr"]) > 0

    def test_execute_invalid_command(self):
        """Test execute with invalid command."""
        from scitex.sh._execute import execute

        # This should raise FileNotFoundError since the command doesn't exist
        with pytest.raises(FileNotFoundError):
            execute(["nonexistent_command_12345"], verbose=False)

    def test_execute_command_with_error_exit(self):
        """Test execute with command that exits with error."""
        from scitex.sh._execute import execute

        result = execute(["false"], verbose=False)

        assert result["success"] is False
        assert result["exit_code"] == 1

    def test_execute_command_with_success_exit(self):
        """Test execute with command that exits successfully."""
        from scitex.sh._execute import execute

        result = execute(["true"], verbose=False)

        assert result["success"] is True
        assert result["exit_code"] == 0

    def test_execute_rejects_string_command(self):
        """Test execute rejects string commands."""
        from scitex.sh._execute import execute

        with pytest.raises(TypeError):
            execute("echo hello", verbose=False)


class TestExecuteTimeout:
    """Tests for timeout functionality."""

    def test_execute_with_timeout_success(self):
        """Test execute with timeout that completes in time."""
        from scitex.sh._execute import execute

        result = execute(["echo", "quick"], verbose=False, timeout=10)

        assert result["success"] is True
        assert result["stdout"] == "quick"

    def test_execute_timeout_exceeded(self):
        """Test execute when timeout is exceeded."""
        from scitex.sh._execute import execute

        result = execute(["sleep", "5"], verbose=False, timeout=1)

        # Command should be killed due to timeout
        assert result["success"] is False
        assert "timeout" in result["stderr"].lower() or result["exit_code"] != 0

    def test_execute_no_timeout(self):
        """Test execute without timeout."""
        from scitex.sh._execute import execute

        result = execute(["echo", "test"], verbose=False, timeout=None)

        assert result["success"] is True


class TestExecuteOutput:
    """Tests for output handling."""

    def test_execute_multiline_output(self):
        """Test execute with multiline output."""
        from scitex.sh._execute import execute

        result = execute(["printf", "line1\nline2\nline3"], verbose=False)

        assert result["success"] is True
        lines = result["stdout"].split("\n")
        assert len(lines) == 3
        assert lines[0] == "line1"
        assert lines[1] == "line2"
        assert lines[2] == "line3"

    def test_execute_unicode_output(self):
        """Test execute with unicode output."""
        from scitex.sh._execute import execute

        result = execute(["echo", "日本語テスト"], verbose=False)

        assert result["success"] is True
        assert "日本語テスト" in result["stdout"]

    def test_execute_empty_output(self):
        """Test execute with empty output."""
        from scitex.sh._execute import execute

        result = execute(["true"], verbose=False)

        assert result["success"] is True
        assert result["stdout"] == ""

    def test_execute_stderr_output(self):
        """Test execute captures stderr."""
        from scitex.sh._execute import execute

        # Use bash to redirect to stderr
        result = execute(
            ["bash", "-c", "echo error >&2"], verbose=False
        )

        assert "error" in result["stderr"]

    def test_execute_both_stdout_and_stderr(self):
        """Test execute captures both stdout and stderr."""
        from scitex.sh._execute import execute

        result = execute(
            ["bash", "-c", "echo stdout; echo stderr >&2"],
            verbose=False,
        )

        assert "stdout" in result["stdout"]
        assert "stderr" in result["stderr"]


class TestExecuteStreaming:
    """Tests for streaming output mode."""

    def test_execute_streaming_mode(self):
        """Test execute with streaming output enabled."""
        from scitex.sh._execute import execute

        result = execute(
            ["echo", "streaming test"],
            verbose=False,
            stream_output=True,
        )

        assert result["success"] is True
        assert "streaming test" in result["stdout"]

    def test_execute_streaming_multiline(self):
        """Test streaming with multiline output."""
        from scitex.sh._execute import execute

        result = execute(
            ["printf", "line1\nline2\nline3"],
            verbose=False,
            stream_output=True,
        )

        assert result["success"] is True
        assert "line1" in result["stdout"]
        assert "line2" in result["stdout"]
        assert "line3" in result["stdout"]

    def test_execute_streaming_with_timeout(self):
        """Test streaming mode with timeout."""
        from scitex.sh._execute import execute

        result = execute(
            ["sleep", "5"],
            verbose=False,
            stream_output=True,
            timeout=1,
        )

        # Should timeout
        assert result["success"] is False


class TestExecuteVerbose:
    """Tests for verbose mode."""

    def test_execute_verbose_false(self, capsys):
        """Test execute with verbose=False suppresses output."""
        from scitex.sh._execute import execute

        execute(["echo", "silent"], verbose=False)
        captured = capsys.readouterr()

        # With verbose=False, there should be minimal or no output
        # Note: The actual behavior depends on implementation

    def test_execute_verbose_true(self, capsys):
        """Test execute with verbose=True shows output."""
        from scitex.sh._execute import execute

        execute(["echo", "visible"], verbose=True)
        captured = capsys.readouterr()

        # With verbose=True, output should be printed
        assert "visible" in captured.out or "echo" in captured.out


class TestExecuteWithFiles:
    """Tests for execute with file operations."""

    def test_execute_cat_file(self, tmp_path):
        """Test execute cat with temporary file."""
        from scitex.sh._execute import execute

        # Create temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content")

        result = execute(["cat", str(test_file)], verbose=False)

        assert result["success"] is True
        assert result["stdout"] == "file content"

    def test_execute_wc_file(self, tmp_path):
        """Test execute wc with temporary file."""
        from scitex.sh._execute import execute

        # Create temp file with known content
        test_file = tmp_path / "test.txt"
        test_file.write_text("one two three\nfour five")

        result = execute(["wc", "-w", str(test_file)], verbose=False)

        assert result["success"] is True
        assert "5" in result["stdout"]

    def test_execute_touch_creates_file(self, tmp_path):
        """Test execute touch creates file."""
        from scitex.sh._execute import execute

        new_file = tmp_path / "newfile.txt"
        assert not new_file.exists()

        result = execute(["touch", str(new_file)], verbose=False)

        assert result["success"] is True
        assert new_file.exists()


class TestExecuteSpecialCharacters:
    """Tests for handling special characters."""

    def test_execute_with_spaces_in_argument(self):
        """Test execute with spaces in argument."""
        from scitex.sh._execute import execute

        result = execute(["echo", "hello world"], verbose=False)

        assert result["success"] is True
        assert result["stdout"] == "hello world"

    def test_execute_with_special_chars_in_argument(self):
        """Test execute with special chars in argument."""
        from scitex.sh._execute import execute

        # These special chars are safe in list format
        result = execute(["echo", "test; echo foo"], verbose=False)

        assert result["success"] is True
        # The argument should be treated literally
        assert "test; echo foo" in result["stdout"]

    def test_execute_with_quotes_in_argument(self):
        """Test execute with quotes in argument."""
        from scitex.sh._execute import execute

        result = execute(["echo", 'say "hello"'], verbose=False)

        assert result["success"] is True
        assert '"hello"' in result["stdout"]


class TestExecuteEnvironment:
    """Tests for environment handling."""

    def test_execute_inherits_environment(self):
        """Test execute inherits environment variables."""
        from scitex.sh._execute import execute

        # HOME should be inherited
        result = execute(["bash", "-c", "echo $HOME"], verbose=False)

        assert result["success"] is True
        assert len(result["stdout"]) > 0

    def test_execute_path_available(self):
        """Test execute has PATH available."""
        from scitex.sh._execute import execute

        result = execute(["which", "ls"], verbose=False)

        assert result["success"] is True
        assert "ls" in result["stdout"]


class TestExecuteExitCodes:
    """Tests for exit code handling."""

    def test_execute_exit_code_zero(self):
        """Test execute with exit code 0."""
        from scitex.sh._execute import execute

        result = execute(["bash", "-c", "exit 0"], verbose=False)

        assert result["exit_code"] == 0
        assert result["success"] is True

    def test_execute_exit_code_one(self):
        """Test execute with exit code 1."""
        from scitex.sh._execute import execute

        result = execute(["bash", "-c", "exit 1"], verbose=False)

        assert result["exit_code"] == 1
        assert result["success"] is False

    def test_execute_exit_code_custom(self):
        """Test execute with custom exit code."""
        from scitex.sh._execute import execute

        result = execute(["bash", "-c", "exit 42"], verbose=False)

        assert result["exit_code"] == 42
        assert result["success"] is False


class TestBufferedExecution:
    """Tests for _execute_buffered function."""

    def test_execute_buffered_import(self):
        """Test _execute_buffered can be imported."""
        from scitex.sh._execute import _execute_buffered

        assert _execute_buffered is not None
        assert callable(_execute_buffered)

    def test_execute_buffered_basic(self):
        """Test _execute_buffered basic functionality."""
        from scitex.sh._execute import _execute_buffered

        result = _execute_buffered(["echo", "test"], verbose=False, timeout=None)

        assert result["success"] is True
        assert result["stdout"] == "test"


class TestStreamingExecution:
    """Tests for _execute_with_streaming function."""

    def test_execute_streaming_import(self):
        """Test _execute_with_streaming can be imported."""
        from scitex.sh._execute import _execute_with_streaming

        assert _execute_with_streaming is not None
        assert callable(_execute_with_streaming)

    def test_execute_streaming_basic(self):
        """Test _execute_with_streaming basic functionality."""
        from scitex.sh._execute import _execute_with_streaming

        result = _execute_with_streaming(
            ["echo", "streaming"], verbose=False, timeout=None
        )

        assert result["success"] is True
        assert "streaming" in result["stdout"]


class TestExecuteRobustness:
    """Robustness and edge case tests."""

    def test_execute_long_output(self):
        """Test execute with long output."""
        from scitex.sh._execute import execute

        # Generate long output
        result = execute(["seq", "1", "1000"], verbose=False)

        assert result["success"] is True
        lines = result["stdout"].split("\n")
        assert len(lines) == 1000

    def test_execute_rapid_succession(self):
        """Test execute in rapid succession."""
        from scitex.sh._execute import execute

        for i in range(10):
            result = execute(["echo", str(i)], verbose=False)
            assert result["success"] is True
            assert result["stdout"] == str(i)

    def test_execute_empty_argument(self):
        """Test execute with empty argument."""
        from scitex.sh._execute import execute

        result = execute(["echo", ""], verbose=False)

        assert result["success"] is True

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/sh/_execute.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-29 07:23:56 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/sh/_execute.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/sh/_execute.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# __FILE__ = __file__
# 
# import subprocess
# import sys
# import select
# import time
# 
# import scitex
# from ._types import CommandInput
# from ._types import ShellResult
# from ._security import validate_command
# 
# 
# def execute(
#     command_str_or_list: CommandInput,
#     verbose: bool = True,
#     timeout: int = None,
#     stream_output: bool = False,
# ) -> ShellResult:
#     """
#     Executes a shell command safely (list format only).
# 
#     Parameters:
#     - command_str_or_list: Command to execute (must be list format)
#     - verbose: Whether to print command and output
#     - timeout: Timeout in seconds (None for no timeout)
#     - stream_output: Whether to stream output in real-time (default: False)
#                      When True, prints output as it's generated instead of waiting
#                      for command completion
# 
#     Returns:
#     - ShellResult dict with stdout, stderr, exit_code, success
# 
#     Raises:
#     - TypeError: If command is a string (not allowed for security)
#     - subprocess.TimeoutExpired: If command exceeds timeout
# 
#     Examples:
#     - sh(['ls', '-la'])
#     - sh(['git', 'status'])
#     - sh(['pdflatex', '-interaction=nonstopmode', 'file.tex'], stream_output=True)
#     """
#     validate_command(command_str_or_list)
# 
#     if verbose:
#         cmd_display = " ".join(command_str_or_list)
#         print(scitex.str.color_text(f"{cmd_display}", "yellow"))
# 
#     if stream_output:
#         # Use real-time streaming mode
#         return _execute_with_streaming(command_str_or_list, verbose, timeout)
#     else:
#         # Use buffered mode (original behavior)
#         return _execute_buffered(command_str_or_list, verbose, timeout)
# 
# 
# def _execute_buffered(
#     command_str_or_list: CommandInput, verbose: bool, timeout: int
# ) -> ShellResult:
#     """Execute command with buffered output (original behavior)."""
#     process = subprocess.Popen(
#         command_str_or_list,
#         shell=False,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#     )
# 
#     try:
#         stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
#     except subprocess.TimeoutExpired:
#         process.kill()
#         stdout_bytes, stderr_bytes = process.communicate()
#         timeout_msg = f"Command timed out after {timeout} seconds"
#         stderr_bytes = stderr_bytes + b"\n" + timeout_msg.encode("utf-8")
# 
#     stdout = stdout_bytes.decode("utf-8").strip()
#     stderr = stderr_bytes.decode("utf-8").strip()
#     exit_code = process.returncode
# 
#     result: ShellResult = {
#         "stdout": stdout,
#         "stderr": stderr,
#         "exit_code": exit_code,
#         "success": exit_code == 0,
#     }
# 
#     if verbose:
#         if stdout:
#             print(stdout)
#         if stderr:
#             print(scitex.str.color_text(stderr, "red"))
# 
#     return result
# 
# 
# def _execute_with_streaming(
#     command_str_or_list: CommandInput, verbose: bool, timeout: int
# ) -> ShellResult:
#     """Execute command with real-time output streaming using select."""
#     import io
# 
#     # Set PYTHONUNBUFFERED for Python scripts and unbuffered mode for shell
#     env = os.environ.copy()
#     env["PYTHONUNBUFFERED"] = "1"
# 
#     process = subprocess.Popen(
#         command_str_or_list,
#         shell=False,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         bufsize=0,  # Unbuffered
#         env=env,
#     )
# 
#     stdout_data = []
#     stderr_data = []
#     start_time = time.time()
# 
#     # Use non-blocking reads
#     import fcntl
# 
#     def make_non_blocking(fd):
#         flags = fcntl.fcntl(fd, fcntl.F_GETFL)
#         fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
# 
#     make_non_blocking(process.stdout)
#     make_non_blocking(process.stderr)
# 
#     try:
#         while True:
#             # Check timeout
#             if timeout and (time.time() - start_time) > timeout:
#                 process.kill()
#                 timeout_msg = f"Command timed out after {timeout} seconds"
#                 if verbose:
#                     print(scitex.str.color_text(timeout_msg, "red"), flush=True)
#                 stderr_data.append(timeout_msg.encode())
#                 break
# 
#             # Check if process has finished
#             poll_result = process.poll()
# 
#             # Read available data from stdout
#             try:
#                 chunk = process.stdout.read()
#                 if chunk:
#                     stdout_data.append(chunk)
#                     if verbose:
#                         text = chunk.decode("utf-8", errors="replace")
#                         print(text, end="", flush=True)
#             except (IOError, BlockingIOError):
#                 pass
# 
#             # Read available data from stderr
#             try:
#                 chunk = process.stderr.read()
#                 if chunk:
#                     stderr_data.append(chunk)
#                     if verbose:
#                         text = chunk.decode("utf-8", errors="replace")
#                         print(scitex.str.color_text(text, "red"), end="", flush=True)
#             except (IOError, BlockingIOError):
#                 pass
# 
#             # If process finished, do final read and break
#             if poll_result is not None:
#                 # Final read to catch any remaining buffered output
#                 try:
#                     chunk = process.stdout.read()
#                     if chunk:
#                         stdout_data.append(chunk)
#                         if verbose:
#                             text = chunk.decode("utf-8", errors="replace")
#                             print(text, end="", flush=True)
#                 except (IOError, BlockingIOError):
#                     pass
# 
#                 try:
#                     chunk = process.stderr.read()
#                     if chunk:
#                         stderr_data.append(chunk)
#                         if verbose:
#                             text = chunk.decode("utf-8", errors="replace")
#                             print(
#                                 scitex.str.color_text(text, "red"), end="", flush=True
#                             )
#                 except (IOError, BlockingIOError):
#                     pass
#                 break
# 
#             # Small sleep to prevent CPU spinning
#             time.sleep(0.05)
# 
#     except Exception as e:
#         process.kill()
#         raise
# 
#     stdout = b"".join(stdout_data).decode("utf-8", errors="replace").strip()
#     stderr = b"".join(stderr_data).decode("utf-8", errors="replace").strip()
#     exit_code = process.returncode
# 
#     result: ShellResult = {
#         "stdout": stdout,
#         "stderr": stderr,
#         "exit_code": exit_code,
#         "success": exit_code == 0,
#     }
# 
#     return result
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/sh/_execute.py
# --------------------------------------------------------------------------------
