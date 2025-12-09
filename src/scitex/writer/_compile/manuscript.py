#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/manuscript.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_compile/manuscript.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Manuscript compilation function.

Provides manuscript-specific compilation with options for:
- Figure exclusion for quick compilation
- PowerPoint to TIF conversion
- TIF cropping
- Verbose/quiet modes
- Force recompilation
"""

from pathlib import Path
from typing import Optional, Callable
from ._runner import run_compile
from ..dataclasses import CompilationResult


def compile_manuscript(
    project_dir: Path,
    timeout: int = 300,
    no_figs: bool = False,
    ppt2tif: bool = False,
    crop_tif: bool = False,
    quiet: bool = False,
    verbose: bool = False,
    force: bool = False,
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
    no_figs : bool
        Exclude figures for quick compilation
    ppt2tif : bool
        Convert PowerPoint to TIF on WSL
    crop_tif : bool
        Crop TIF images to remove excess whitespace
    quiet : bool
        Suppress detailed logs for LaTeX compilation
    verbose : bool
        Show detailed logs for LaTeX compilation
    force : bool
        Force full recompilation, ignore cache
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
    >>> from scitex.writer._compile import compile_manuscript
    >>>
    >>> # Quick compilation without figures
    >>> result = compile_manuscript(
    ...     project_dir=Path("~/my-paper"),
    ...     no_figs=True,
    ...     quiet=True
    ... )
    >>>
    >>> # Full compilation with PowerPoint conversion
    >>> result = compile_manuscript(
    ...     project_dir=Path("~/my-paper"),
    ...     ppt2tif=True,
    ...     crop_tif=True,
    ...     verbose=True
    ... )
    """
    return run_compile(
        "manuscript",
        project_dir,
        timeout=timeout,
        no_figs=no_figs,
        ppt2tif=ppt2tif,
        crop_tif=crop_tif,
        quiet=quiet,
        verbose=verbose,
        force=force,
        log_callback=log_callback,
        progress_callback=progress_callback,
    )


__all__ = ["compile_manuscript"]

# EOF
