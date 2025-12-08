#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 07:23:56 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/sh/_execute.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/sh/_execute.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

__FILE__ = __file__

import subprocess
import sys
import select
import time

import scitex
from ._types import CommandInput
from ._types import ShellResult
from ._security import validate_command


def execute(
    command_str_or_list: CommandInput,
    verbose: bool = True,
    timeout: int = None,
    stream_output: bool = False,
) -> ShellResult:
    """
    Executes a shell command safely (list format only).

    Parameters:
    - command_str_or_list: Command to execute (must be list format)
    - verbose: Whether to print command and output
    - timeout: Timeout in seconds (None for no timeout)
    - stream_output: Whether to stream output in real-time (default: False)
                     When True, prints output as it's generated instead of waiting
                     for command completion

    Returns:
    - ShellResult dict with stdout, stderr, exit_code, success

    Raises:
    - TypeError: If command is a string (not allowed for security)
    - subprocess.TimeoutExpired: If command exceeds timeout

    Examples:
    - sh(['ls', '-la'])
    - sh(['git', 'status'])
    - sh(['pdflatex', '-interaction=nonstopmode', 'file.tex'], stream_output=True)
    """
    validate_command(command_str_or_list)

    if verbose:
        cmd_display = " ".join(command_str_or_list)
        print(scitex.str.color_text(f"{cmd_display}", "yellow"))

    if stream_output:
        # Use real-time streaming mode
        return _execute_with_streaming(command_str_or_list, verbose, timeout)
    else:
        # Use buffered mode (original behavior)
        return _execute_buffered(command_str_or_list, verbose, timeout)


def _execute_buffered(
    command_str_or_list: CommandInput, verbose: bool, timeout: int
) -> ShellResult:
    """Execute command with buffered output (original behavior)."""
    process = subprocess.Popen(
        command_str_or_list,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_bytes, stderr_bytes = process.communicate()
        timeout_msg = f"Command timed out after {timeout} seconds"
        stderr_bytes = stderr_bytes + b"\n" + timeout_msg.encode("utf-8")

    stdout = stdout_bytes.decode("utf-8").strip()
    stderr = stderr_bytes.decode("utf-8").strip()
    exit_code = process.returncode

    result: ShellResult = {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "success": exit_code == 0,
    }

    if verbose:
        if stdout:
            print(stdout)
        if stderr:
            print(scitex.str.color_text(stderr, "red"))

    return result


def _execute_with_streaming(
    command_str_or_list: CommandInput, verbose: bool, timeout: int
) -> ShellResult:
    """Execute command with real-time output streaming using select."""
    import io

    # Set PYTHONUNBUFFERED for Python scripts and unbuffered mode for shell
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command_str_or_list,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,  # Unbuffered
        env=env,
    )

    stdout_data = []
    stderr_data = []
    start_time = time.time()

    # Use non-blocking reads
    import fcntl

    def make_non_blocking(fd):
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    make_non_blocking(process.stdout)
    make_non_blocking(process.stderr)

    try:
        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                process.kill()
                timeout_msg = f"Command timed out after {timeout} seconds"
                if verbose:
                    print(scitex.str.color_text(timeout_msg, "red"), flush=True)
                stderr_data.append(timeout_msg.encode())
                break

            # Check if process has finished
            poll_result = process.poll()

            # Read available data from stdout
            try:
                chunk = process.stdout.read()
                if chunk:
                    stdout_data.append(chunk)
                    if verbose:
                        text = chunk.decode("utf-8", errors="replace")
                        print(text, end="", flush=True)
            except (IOError, BlockingIOError):
                pass

            # Read available data from stderr
            try:
                chunk = process.stderr.read()
                if chunk:
                    stderr_data.append(chunk)
                    if verbose:
                        text = chunk.decode("utf-8", errors="replace")
                        print(scitex.str.color_text(text, "red"), end="", flush=True)
            except (IOError, BlockingIOError):
                pass

            # If process finished, do final read and break
            if poll_result is not None:
                # Final read to catch any remaining buffered output
                try:
                    chunk = process.stdout.read()
                    if chunk:
                        stdout_data.append(chunk)
                        if verbose:
                            text = chunk.decode("utf-8", errors="replace")
                            print(text, end="", flush=True)
                except (IOError, BlockingIOError):
                    pass

                try:
                    chunk = process.stderr.read()
                    if chunk:
                        stderr_data.append(chunk)
                        if verbose:
                            text = chunk.decode("utf-8", errors="replace")
                            print(
                                scitex.str.color_text(text, "red"), end="", flush=True
                            )
                except (IOError, BlockingIOError):
                    pass
                break

            # Small sleep to prevent CPU spinning
            time.sleep(0.05)

    except Exception as e:
        process.kill()
        raise

    stdout = b"".join(stdout_data).decode("utf-8", errors="replace").strip()
    stderr = b"".join(stderr_data).decode("utf-8", errors="replace").strip()
    exit_code = process.returncode

    result: ShellResult = {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "success": exit_code == 0,
    }

    return result


# EOF
