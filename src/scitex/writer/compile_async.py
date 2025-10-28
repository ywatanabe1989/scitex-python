#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/compile_async.py

"""
Asynchronous compilation support for writer module.

Allows non-blocking compilation operations for concurrent workflows.
"""

import asyncio
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from scitex.logging import getLogger

from .compile import (
    compile_manuscript,
    compile_supplementary,
    compile_revision,
    CompilationResult
)

logger = getLogger(__name__)

# Thread pool for async operations
_executor = ThreadPoolExecutor(max_workers=2)


async def compile_manuscript_async(
    project_dir: Path,
    additional_args: List[str] = None,
    timeout: int = 300
) -> CompilationResult:
    """
    Asynchronously compile manuscript.

    Non-blocking version of compile_manuscript for use in async contexts.

    Args:
        project_dir: Path to writer project
        additional_args: Additional compilation arguments
        timeout: Timeout in seconds

    Returns:
        CompilationResult

    Example:
        >>> result = await compile_manuscript_async(Path("my_paper"))
        >>> if result.success:
        ...     print(f"PDF: {result.output_pdf}")
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        compile_manuscript,
        Path(project_dir),
        additional_args,
        timeout
    )


async def compile_supplementary_async(
    project_dir: Path,
    additional_args: List[str] = None,
    timeout: int = 300
) -> CompilationResult:
    """
    Asynchronously compile supplementary materials.

    Non-blocking version of compile_supplementary for use in async contexts.

    Args:
        project_dir: Path to writer project
        additional_args: Additional compilation arguments
        timeout: Timeout in seconds

    Returns:
        CompilationResult

    Example:
        >>> result = await compile_supplementary_async(Path("my_paper"))
        >>> if result.success:
        ...     print(f"PDF: {result.output_pdf}")
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        compile_supplementary,
        Path(project_dir),
        additional_args,
        timeout
    )


async def compile_revision_async(
    project_dir: Path,
    track_changes: bool = False,
    additional_args: List[str] = None,
    timeout: int = 300
) -> CompilationResult:
    """
    Asynchronously compile revision response.

    Non-blocking version of compile_revision for use in async contexts.

    Args:
        project_dir: Path to writer project
        track_changes: Whether to track changes
        additional_args: Additional compilation arguments
        timeout: Timeout in seconds

    Returns:
        CompilationResult

    Example:
        >>> result = await compile_revision_async(Path("my_paper"))
        >>> if result.success:
        ...     print(f"PDF: {result.output_pdf}")
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        compile_revision,
        Path(project_dir),
        track_changes,
        additional_args,
        timeout
    )


async def compile_all_async(
    project_dir: Path,
    track_changes: bool = False,
    timeout: int = 300
) -> dict:
    """
    Compile all document types concurrently.

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
            return_exceptions=True
        )
    except Exception as e:
        logger.error(f"Error during concurrent compilation: {e}")
        return {
            'manuscript': None,
            'supplementary': None,
            'revision': None,
        }

    # Handle exceptions and convert to None
    def safe_result(result):
        if isinstance(result, Exception):
            logger.error(f"Compilation exception: {result}")
            return None
        return result

    return {
        'manuscript': safe_result(results[0]),
        'supplementary': safe_result(results[1]),
        'revision': safe_result(results[2]),
    }


__all__ = [
    'compile_manuscript_async',
    'compile_supplementary_async',
    'compile_revision_async',
    'compile_all_async',
]

# EOF
