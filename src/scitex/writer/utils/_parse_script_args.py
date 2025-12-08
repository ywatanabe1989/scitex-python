#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 17:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_parse_script_args.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_parse_script_args.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Script argument parser - extract available arguments from shell scripts.

Parses shell scripts to extract documented arguments from usage() functions.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScriptArgument:
    """Represents a single script argument."""

    short_flag: Optional[str]  # e.g., "-nf"
    long_flag: Optional[str]  # e.g., "--no_figs"
    description: str  # e.g., "Exclude figures for quick compilation"
    default: Optional[str]  # e.g., "false"

    def __str__(self) -> str:
        """Format as help text."""
        flags = ", ".join(f for f in [self.short_flag, self.long_flag] if f)
        default_str = f" (default: {self.default})" if self.default else ""
        return f"{flags:20} {self.description}{default_str}"


class ScriptArgumentParser:
    """Extract arguments from shell script usage() functions."""

    @staticmethod
    def parse(script_path: Path) -> list[ScriptArgument]:
        """
        Parse shell script to extract available arguments.

        Looks for usage() function pattern:
            -flag,  --long-flag  Description (default: value)

        Args:
            script_path: Path to shell script

        Returns:
            List of ScriptArgument objects
        """
        if not script_path.exists():
            return []

        content = script_path.read_text()

        # Find usage() function
        usage_match = re.search(r"usage\s*\(\)\s*{(.*?)exit\s+0", content, re.DOTALL)

        if not usage_match:
            return []

        usage_text = usage_match.group(1)

        # Extract arguments from usage text (contains echo statements)
        # Find all lines with option flags (start with "echo" and contain flags)
        options_text = usage_text
        args = []

        # Parse each line in usage (will contain echo statements)
        for line in options_text.split("\n"):
            # Extract content between quotes in echo statements
            quote_match = re.search(r'"([^"]*)"', line)
            if not quote_match:
                continue

            line_content = quote_match.group(1).strip()
            if not line_content or line_content.startswith("#"):
                continue

            # Skip lines without flags
            if "-" not in line_content:
                continue

            arg = ScriptArgumentParser._parse_argument_line(line_content)
            if arg:
                args.append(arg)

        return args

    @staticmethod
    def _parse_argument_line(line: str) -> Optional[ScriptArgument]:
        """
        Parse single argument line.

        Format: "-nf,  --no_figs       Description (default: value)"
        """
        # Extract flags and description
        flags_match = re.match(r"(.*?)\s{2,}(.*)", line)
        if not flags_match:
            return None

        flags_str = flags_match.group(1).strip()
        rest = flags_match.group(2).strip()

        # Parse flags
        short_flag = None
        long_flag = None

        if "," in flags_str:
            parts = flags_str.split(",")
            short_flag = parts[0].strip() if parts[0].strip() else None
            long_flag = (
                parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            )
        else:
            # Single flag (short or long)
            if flags_str.startswith("--"):
                long_flag = flags_str
            elif flags_str.startswith("-"):
                short_flag = flags_str
            else:
                return None

        # Parse description and default
        description = rest
        default = None

        default_match = re.search(r"\(default:\s*([^)]+)\)", rest)
        if default_match:
            default = default_match.group(1).strip()
            description = rest[: default_match.start()].strip()

        if not description:
            return None

        return ScriptArgument(
            short_flag=short_flag,
            long_flag=long_flag,
            description=description,
            default=default,
        )


__all__ = [
    "ScriptArgument",
    "ScriptArgumentParser",
]

# EOF
