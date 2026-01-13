#!/usr/bin/env python3
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile_unified.py

"""
Unified compilation interface for writer module.

Provides a single `compile()` function that handles all document types
with optional async support.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal, Optional, Union

from .._dataclasses import CompilationResult
from ._compile_async import (
    compile_all_async,
    compile_manuscript_async,
    compile_revision_async,
    compile_supplementary_async,
)
from .manuscript import compile_manuscript
from .revision import compile_revision
from .supplementary import compile_supplementary

DocType = Literal["manuscript", "supplementary", "revision", "all"]


def compile(
    *doc_types: DocType,
    project_dir: Optional[Path] = None,
    async_: bool = False,
    track_changes: bool = False,
    timeout: int = 300,
) -> Union[CompilationResult, dict, asyncio.coroutine]:
    """
    Unified compilation function for LaTeX documents.

    Args:
        *doc_types: Document types to compile. One or more of:
            - "manuscript": Main document
            - "supplementary": Supplementary materials
            - "revision": Revision response
            - "all": All document types (async only)
        project_dir: Path to writer project directory
        async_: If True, returns awaitable coroutine
        track_changes: Enable change tracking for revision (default: False)
        timeout: Compilation timeout in seconds (default: 300)

    Returns
    -------
        - Single doc_type (sync): CompilationResult
        - Multiple doc_types (sync): dict of {doc_type: CompilationResult}
        - async_=True: Awaitable coroutine returning above

    Examples
    --------
        >>> # Sync single document
        >>> result = compile("manuscript", project_dir=Path("."))

        >>> # Sync multiple documents
        >>> results = compile("manuscript", "supplementary", project_dir=Path("."))
        >>> results["manuscript"].success

        >>> # Async single document
        >>> result = await compile("manuscript", project_dir=Path("."), async_=True)

        >>> # Async all documents (parallel)
        >>> results = await compile("all", project_dir=Path("."), async_=True)
    """
    if not doc_types:
        raise ValueError("At least one document type required")

    if project_dir is None:
        raise ValueError("project_dir is required")

    project_dir = Path(project_dir)

    # Map doc types to functions
    sync_funcs = {
        "manuscript": lambda: compile_manuscript(project_dir, timeout=timeout),
        "supplementary": lambda: compile_supplementary(project_dir, timeout=timeout),
        "revision": lambda: compile_revision(
            project_dir, track_changes, timeout=timeout
        ),
    }

    async_funcs = {
        "manuscript": lambda: compile_manuscript_async(project_dir, timeout=timeout),
        "supplementary": lambda: compile_supplementary_async(
            project_dir, timeout=timeout
        ),
        "revision": lambda: compile_revision_async(
            project_dir, track_changes, timeout=timeout
        ),
    }

    # Handle "all" special case
    if "all" in doc_types:
        if async_:
            return compile_all_async(
                project_dir, track_changes=track_changes, timeout=timeout
            )
        else:
            # Sync "all" - compile sequentially
            return {
                "manuscript": sync_funcs["manuscript"](),
                "supplementary": sync_funcs["supplementary"](),
                "revision": sync_funcs["revision"](),
            }

    # Validate doc types
    valid_types = {"manuscript", "supplementary", "revision"}
    for dt in doc_types:
        if dt not in valid_types:
            raise ValueError(
                f"Invalid document type: {dt}. Must be one of {valid_types}"
            )

    # Single document
    if len(doc_types) == 1:
        doc_type = doc_types[0]
        if async_:
            return async_funcs[doc_type]()
        else:
            return sync_funcs[doc_type]()

    # Multiple documents
    if async_:

        async def _compile_multiple():
            tasks = [async_funcs[dt]() for dt in doc_types]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {
                dt: (None if isinstance(r, Exception) else r)
                for dt, r in zip(doc_types, results)
            }

        return _compile_multiple()
    else:
        return {dt: sync_funcs[dt]() for dt in doc_types}


__all__ = ["compile"]

# EOF
