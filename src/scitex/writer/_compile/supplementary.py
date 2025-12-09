#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/supplementary.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_compile/supplementary.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Supplementary materials compilation function.

Provides supplementary-specific compilation with options for:
- Figure inclusion (default)
- PowerPoint to TIF conversion
- TIF cropping
- Quiet mode
"""

from pathlib import Path
from typing import Optional, Callable
from ._runner import run_compile
from ..dataclasses import CompilationResult


def compile_supplementary(
    project_dir: Path,
    timeout: int = 300,
    no_figs: bool = False,
    ppt2tif: bool = False,
    crop_tif: bool = False,
    quiet: bool = False,
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
    no_figs : bool
        Exclude figures (default includes figures)
    ppt2tif : bool
        Convert PowerPoint to TIF on WSL
    crop_tif : bool
        Crop TIF images to remove excess whitespace
    quiet : bool
        Suppress detailed logs for LaTeX compilation
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
    >>> from scitex.writer._compile import compile_supplementary
    >>>
    >>> # Standard compilation with figures
    >>> result = compile_supplementary(
    ...     project_dir=Path("~/my-paper"),
    ...     ppt2tif=True,
    ...     quiet=False
    ... )
    >>>
    >>> # Quick compilation without figures
    >>> result = compile_supplementary(
    ...     project_dir=Path("~/my-paper"),
    ...     no_figs=True,
    ...     quiet=True
    ... )
    """
    return run_compile(
        "supplementary",
        project_dir,
        timeout=timeout,
        no_figs=no_figs,
        ppt2tif=ppt2tif,
        crop_tif=crop_tif,
        quiet=quiet,
        log_callback=log_callback,
        progress_callback=progress_callback,
    )


__all__ = ["compile_supplementary"]

# EOF
