#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 07:23:56 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/sh/_execute.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/sh/_execute.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

__FILE__ = __file__

import subprocess

import scitex
from ._types import CommandInput
from ._types import ShellResult
from ._security import validate_command


def execute(
    command_str_or_list: CommandInput, verbose: bool = True
) -> ShellResult:
    """
    Executes a shell command safely (list format only).

    Parameters:
    - command_str_or_list: Command to execute (must be list format)
    - verbose: Whether to print command and output

    Returns:
    - ShellResult dict with stdout, stderr, exit_code, success

    Raises:
    - TypeError: If command is a string (not allowed for security)

    Examples:
    - sh(['ls', '-la'])
    - sh(['git', 'status'])
    - sh(['pdflatex', '-interaction=nonstopmode', 'file.tex'])
    """
    validate_command(command_str_or_list)

    if verbose:
        cmd_display = " ".join(command_str_or_list)
        print(scitex.str.color_text(f"{cmd_display}", "yellow"))

    process = subprocess.Popen(
        command_str_or_list,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_bytes, stderr_bytes = process.communicate()

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

# EOF
