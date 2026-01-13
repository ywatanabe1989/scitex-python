#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/scholar/jobs/__init__.py
# ----------------------------------------

"""Async job management for Scholar operations.

This module provides non-blocking job execution for long-running tasks
like PDF fetching, metadata enrichment, and batch operations.

Example:
    from scitex.scholar.jobs import get_job_manager, JobType

    manager = get_job_manager()

    # Submit a fetch job (returns immediately)
    job_id = await manager.submit_async(
        job_type=JobType.FETCH,
        params={"dois": ["10.1038/nature12373"]},
        executor=fetch_executor,
    )

    # Check status
    status = manager.get_status(job_id)
    print(f"Progress: {status['progress']['percent']}%")

    # Wait for completion
    result = await manager.wait_for_job(job_id)
"""

from ._errors import (
    ErrorType,
    StructuredError,
    categorize_error,
    collect_recent_screenshots,
    create_structured_error,
    get_suggested_action,
    is_recoverable,
)
from ._executors import (
    EXECUTORS,
    enrich_bibtex_executor,
    fetch_bibtex_executor,
    fetch_multiple_executor,
    fetch_single_executor,
    get_executor,
)
from ._Job import Job, JobProgress, JobStatus, JobType
from ._JobManager import JobManager, get_job_manager

__all__ = [
    "Job",
    "JobProgress",
    "JobStatus",
    "JobType",
    "JobManager",
    "get_job_manager",
    "EXECUTORS",
    "get_executor",
    "fetch_single_executor",
    "fetch_multiple_executor",
    "fetch_bibtex_executor",
    "enrich_bibtex_executor",
    # Error handling
    "ErrorType",
    "StructuredError",
    "categorize_error",
    "create_structured_error",
    "collect_recent_screenshots",
    "get_suggested_action",
    "is_recoverable",
]

# EOF
