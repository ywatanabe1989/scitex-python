#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/__init__.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = "./src/scitex/writer/_compile/__init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
LaTeX compilation module for writer.

Provides organized compilation functionality:
- _runner: Script execution
- _parser: Output parsing
- _validator: Pre-compile validation
"""

from pathlib import Path
from typing import Optional, Callable
from ._runner import run_compile
from ..dataclasses import CompilationResult


def compile_manuscript(
    project_dir: Path,
    timeout: int = 300,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> CompilationResult:
    """
    Compile manuscript document with optional callbacks.

    Parameters
    ----------
    project_dir : Path
        Path to writer project directory
    timeout : int
        Timeout in seconds
    log_callback : Optional[Callable[[str], None]]
        Called with each log line
    progress_callback : Optional[Callable[[int, str], None]]
        Called with progress updates (percent, step)

    Returns
    -------
    CompilationResult
        Compilation status and outputs
    """
    return run_compile(
        "manuscript",
        project_dir,
        timeout=timeout,
        log_callback=log_callback,
        progress_callback=progress_callback,
    )


def compile_supplementary(
    project_dir: Path,
    timeout: int = 300,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> CompilationResult:
    """
    Compile supplementary materials with optional callbacks.

    Parameters
    ----------
    project_dir : Path
        Path to writer project directory
    timeout : int
        Timeout in seconds
    log_callback : Optional[Callable[[str], None]]
        Called with each log line
    progress_callback : Optional[Callable[[int, str], None]]
        Called with progress updates (percent, step)

    Returns
    -------
    CompilationResult
        Compilation status and outputs
    """
    return run_compile(
        "supplementary",
        project_dir,
        timeout=timeout,
        log_callback=log_callback,
        progress_callback=progress_callback,
    )


def compile_revision(
    project_dir: Path,
    track_changes: bool = False,
    timeout: int = 300,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> CompilationResult:
    """
    Compile revision responses with optional callbacks.

    Parameters
    ----------
    project_dir : Path
        Path to writer project directory
    track_changes : bool
        Whether to enable change tracking
    timeout : int
        Timeout in seconds
    log_callback : Optional[Callable[[str], None]]
        Called with each log line
    progress_callback : Optional[Callable[[int, str], None]]
        Called with progress updates (percent, step)

    Returns
    -------
    CompilationResult
        Compilation status and outputs
    """
    return run_compile(
        "revision",
        project_dir,
        timeout=timeout,
        track_changes=track_changes,
        log_callback=log_callback,
        progress_callback=progress_callback,
    )


__all__ = [
    "run_compile",
    "compile_manuscript",
    "compile_supplementary",
    "compile_revision",
    "CompilationResult",
]

# EOF
