#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 07:23:58 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/sh/_security.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/sh/_security.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

__FILE__ = __file__

import shlex
from typing import Union
from typing import List


DANGEROUS_CHARS = [";", "|", "&", "$", "`", "\n", ">", "<", "(", ")", "{", "}"]


def validate_command(command_str_or_list: Union[str, List[str]]) -> None:
    """
    Validates command for security issues.

    Parameters:
    - command_str_or_list: Command string or list to validate

    Raises:
    - TypeError: If command is a string (not allowed for security)
    - ValueError: If command contains dangerous characters
    """
    if isinstance(command_str_or_list, str):
        raise TypeError(
            "String commands are not allowed for security reasons. "
            "Use list format: ['command', 'arg1', 'arg2']. "
            "For pipes and redirects, use Python subprocess chaining instead."
        )

    for arg in command_str_or_list:
        if "\0" in str(arg):
            raise ValueError(
                "Command argument contains null byte - potential shell injection attempt"
            )


def quote(arg: str) -> str:
    """
    Safely quotes a string for use in shell commands.

    Parameters:
    - arg: The argument to quote

    Returns:
    - str: Safely quoted string

    Examples:
    --------
    >>> filename = "file; rm -rf /"
    >>> from scitex.sh import sh, quote
    >>> sh(f"cat {quote(filename)}")
    """
    return shlex.quote(arg)


# EOF
