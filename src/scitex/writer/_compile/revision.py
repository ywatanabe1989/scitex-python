#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/revision.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_compile/revision.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Revision response compilation function.

Provides revision-specific compilation with options for:
- Change tracking (diff highlighting)
"""

from pathlib import Path
from typing import Optional, Callable
from ._runner import run_compile
from ..dataclasses import CompilationResult


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
        Whether to enable change tracking (diff highlighting)
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

    Examples
    --------
    >>> from pathlib import Path
    >>> from scitex.writer._compile import compile_revision
    >>>
    >>> # Standard revision compilation
    >>> result = compile_revision(
    ...     project_dir=Path("~/my-paper")
    ... )
    >>>
    >>> # Compilation with change tracking
    >>> result = compile_revision(
    ...     project_dir=Path("~/my-paper"),
    ...     track_changes=True
    ... )
    """
    return run_compile(
        "revision",
        project_dir,
        timeout=timeout,
        track_changes=track_changes,
        log_callback=log_callback,
        progress_callback=progress_callback,
    )


__all__ = ["compile_revision"]

# EOF
