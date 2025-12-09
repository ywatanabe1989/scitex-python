#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

from typing import Union

from ._types import CommandInput, ReturnFormat, ShellResult
from ._security import quote, validate_command
from ._execute import execute


def sh(
    command_str_or_list: CommandInput,
    verbose: bool = True,
    return_as: ReturnFormat = "dict",
    timeout: int = None,
    stream_output: bool = False,
) -> Union[str, ShellResult]:
    """
    Executes a shell command safely (list format only).

    Parameters:
    - command_str_or_list: Command to execute (MUST be list format)
    - verbose: Whether to print command and output
    - return_as: Return format ("dict" or "str")
    - timeout: Timeout in seconds (None for no timeout)
    - stream_output: Whether to stream output in real-time (default: False)

    Returns:
    - If return_as="str": output string
    - If return_as="dict": ShellResult dict

    Security Notes:
    - Only list format is allowed to prevent shell injection
    - Each argument is treated as a literal string
    - For pipes/redirects, use Python subprocess chaining

    Examples:
    --------
    >>> from scitex.sh import sh
    >>> sh(["ls", "-la", "/home"])
    >>> sh(["git", "status"])
    >>> sh(["sleep", "10"], timeout=5)  # Will timeout after 5 seconds
    >>> sh(["./compile.sh"], stream_output=True)  # Stream output in real-time
    >>>
    >>> # For grep-like filtering, use Python:
    >>> result = sh(["ls", "-la"])
    >>> filtered = [l for l in result['stdout'].split('\\n') if '.py' in l]
    """
    result = execute(
        command_str_or_list,
        verbose=verbose,
        timeout=timeout,
        stream_output=stream_output,
    )

    if return_as == "dict":
        return result
    else:
        if result["success"]:
            return result["stdout"]
        else:
            return result["stderr"]


def sh_run(command: CommandInput, verbose: bool = True) -> ShellResult:
    """
    Executes a shell command and returns detailed results.

    Parameters:
    - command: Command to execute (MUST be list format)
    - verbose: Whether to print command and output

    Returns:
    - ShellResult dict with stdout, stderr, exit_code, success

    Examples:
    --------
    >>> from scitex.sh import sh_run
    >>> result = sh_run(["ls", "-la"])
    >>> if result['success']:
    ...     print(result['stdout'])
    """
    return execute(command, verbose=verbose)


__all__ = ["sh", "sh_run", "quote"]

# EOF
