#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 17:43:50 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile_async.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_compile_async.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Asynchronous compilation support for writer module.

Allows non-blocking compilation operations for concurrent workflows.
"""

import asyncio
from functools import wraps
from pathlib import Path
from typing import Callable
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from scitex.logging import getLogger

from ._compile import compile_manuscript
from ._compile import compile_supplementary
from ._compile import compile_revision
from ._compile import CompilationResult

logger = getLogger(__name__)

# Thread pool for async operations
_executor = ThreadPoolExecutor(max_workers=2)


def _make_async_wrapper(sync_func: Callable) -> Callable:
    """
    Factory function to create async wrappers for sync compilation functions.

    Args:
        sync_func: Synchronous compilation function

    Returns:
        Async wrapper function
    """

    @wraps(sync_func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> CompilationResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, sync_func, *args, **kwargs)

    return async_wrapper


# Create async wrappers using factory function
compile_manuscript_async = _make_async_wrapper(compile_manuscript)
compile_supplementary_async = _make_async_wrapper(compile_supplementary)
compile_revision_async = _make_async_wrapper(compile_revision)


async def compile_all_async(
    project_dir: Path, track_changes: bool = False, timeout: int = 300
) -> dict:
    """
    Compile all document dataclasses concurrently.

    Runs all three document compilations in parallel for faster overall completion.

    Args:
        project_dir: Path to writer project
        track_changes: Whether to track changes in revision
        timeout: Timeout per compilation

    Returns:
        Dict with keys 'manuscript', 'supplementary', 'revision' and CompilationResult values.
        If a compilation fails, the value will be None.

    Example:
        >>> results = await compile_all_async(Path("my_paper"))
        >>> for doc_type, result in results.items():
        ...     if result and result.success:
        ...         print(f"{doc_type}: OK")
        ...     else:
        ...         print(f"{doc_type}: FAILED")
    """
    logger.info(f"Starting concurrent compilation of all documents in {project_dir}")

    try:
        results = await asyncio.gather(
            compile_manuscript_async(project_dir, timeout=timeout),
            compile_supplementary_async(project_dir, timeout=timeout),
            compile_revision_async(project_dir, track_changes, timeout=timeout),
            return_exceptions=True,
        )
    except Exception as e:
        logger.error(f"Error during concurrent compilation: {e}")
        return {
            "manuscript": None,
            "supplementary": None,
            "revision": None,
        }

    # Handle exceptions and convert to None
    def safe_result(result):
        if isinstance(result, Exception):
            logger.error(f"Compilation exception: {result}")
            return None
        return result

    return {
        "manuscript": safe_result(results[0]),
        "supplementary": safe_result(results[1]),
        "revision": safe_result(results[2]),
    }


__all__ = [
    "compile_manuscript_async",
    "compile_supplementary_async",
    "compile_revision_async",
    "compile_all_async",
]

# EOF
