#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/scholar/jobs/_JobManager.py
# ----------------------------------------

"""Job manager for async task execution and tracking."""

from __future__ import annotations

import asyncio
import os
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Coroutine

from ._Job import Job, JobStatus, JobType


def _get_jobs_dir() -> Path:
    """Get the jobs directory path."""
    base_dir = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
    jobs_dir = base_dir / "scholar" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir


class JobManager:
    """Manages async job execution and persistence.

    Jobs are stored in ~/.scitex/scholar/jobs/ as JSON files.
    Supports:
    - Submitting new jobs (sync or async)
    - Tracking job progress
    - Cancelling running jobs
    - Listing active and completed jobs
    - Cleaning up old jobs

    Example:
        manager = JobManager()

        # Submit a job (returns immediately)
        job_id = await manager.submit_async(
            job_type="fetch",
            params={"dois": ["10.1038/nature12373"]},
        )

        # Check status
        job = manager.get_job(job_id)
        print(job.status, job.progress.percent)

        # Wait for completion
        result = await manager.wait_for_job(job_id)
    """

    def __init__(self, jobs_dir: Path | None = None):
        """Initialize JobManager.

        Args:
            jobs_dir: Custom jobs directory path (default: ~/.scitex/scholar/jobs/)
        """
        self.jobs_dir = jobs_dir or _get_jobs_dir()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of running jobs
        self._running_jobs: dict[str, Job] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}

    # -------------------------------------------------------------------------
    # Job Submission
    # -------------------------------------------------------------------------

    def submit(
        self,
        job_type: JobType | str,
        params: dict[str, Any],
        job_id: str | None = None,
    ) -> str:
        """Submit a new job (creates job record, doesn't execute).

        Args:
            job_type: Type of job (fetch, enrich, download_pdf, etc.)
            params: Parameters for the job
            job_id: Optional custom job ID

        Returns
        -------
            Job ID
        """
        job = Job(
            job_type=job_type,
            params=params,
        )
        if job_id:
            job.job_id = job_id

        # Save to disk
        job.save(self.jobs_dir)
        return job.job_id

    async def submit_async(
        self,
        job_type: JobType | str,
        params: dict[str, Any],
        executor: Callable[..., Coroutine[Any, Any, dict]] | None = None,
    ) -> str:
        """Submit and start executing a job asynchronously.

        Args:
            job_type: Type of job
            params: Parameters for the job
            executor: Async function to execute the job

        Returns
        -------
            Job ID (job starts executing in background)
        """
        job = Job(job_type=job_type, params=params)
        job.save(self.jobs_dir)

        if executor:
            # Start execution in background
            task = asyncio.create_task(self._execute_job(job, executor))
            self._running_jobs[job.job_id] = job
            self._running_tasks[job.job_id] = task

        return job.job_id

    async def _execute_job(
        self,
        job: Job,
        executor: Callable[..., Coroutine[Any, Any, dict]],
    ) -> None:
        """Execute a job and update its status."""
        try:
            job.start()
            job.save(self.jobs_dir)

            # Execute with progress callback
            result = await executor(
                **job.params,
                progress_callback=lambda **kwargs: self._update_progress(
                    job.job_id, **kwargs
                ),
            )

            job.complete(result)
        except asyncio.CancelledError:
            job.cancel()
        except Exception as e:
            # Create structured error for unexpected exceptions
            from ._errors import create_structured_error

            structured_error = create_structured_error(
                error=e,
                include_screenshots=True,
            )
            job.fail(str(e), structured_error=structured_error.to_dict())
        finally:
            job.save(self.jobs_dir)
            self._running_jobs.pop(job.job_id, None)
            self._running_tasks.pop(job.job_id, None)

    def _update_progress(self, job_id: str, **kwargs) -> None:
        """Update job progress (called by executor)."""
        job = self._running_jobs.get(job_id)
        if job:
            job.update_progress(**kwargs)
            job.save(self.jobs_dir)

    # -------------------------------------------------------------------------
    # Job Retrieval
    # -------------------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID.

        Args:
            job_id: Job ID

        Returns
        -------
            Job object or None if not found
        """
        # Check in-memory cache first (for running jobs)
        if job_id in self._running_jobs:
            return self._running_jobs[job_id]

        # Load from disk
        job_file = self.jobs_dir / f"{job_id}.json"
        if job_file.exists():
            return Job.load(job_file)

        return None

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        """Get job status as dictionary.

        Args:
            job_id: Job ID

        Returns
        -------
            Status dictionary or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "job_type": job.job_type.value
            if isinstance(job.job_type, JobType)
            else job.job_type,
            "status": job.status.value,
            "progress": job.progress.to_dict(),
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "duration_seconds": job.duration_seconds,
            "error": job.error,
            "structured_error": job.structured_error,
            "has_result": job.result is not None,
        }

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        """Get job result.

        Args:
            job_id: Job ID

        Returns
        -------
            Result dictionary or None
        """
        job = self.get_job(job_id)
        if job and job.result:
            return job.result
        return None

    def list_jobs(
        self,
        status: JobStatus | str | None = None,
        job_type: JobType | str | None = None,
        limit: int = 50,
        include_completed: bool = True,
    ) -> list[dict[str, Any]]:
        """List jobs with optional filtering.

        Args:
            status: Filter by status
            job_type: Filter by job type
            limit: Maximum number of jobs to return
            include_completed: Include completed/failed/cancelled jobs

        Returns
        -------
            List of job status dictionaries
        """
        jobs = []

        # Load all jobs from disk
        for job_file in sorted(
            self.jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                job = Job.load(job_file)

                # Apply filters
                if status:
                    status_val = (
                        status.value if isinstance(status, JobStatus) else status
                    )
                    if job.status.value != status_val:
                        continue

                if job_type:
                    type_val = (
                        job_type.value if isinstance(job_type, JobType) else job_type
                    )
                    job_type_val = (
                        job.job_type.value
                        if isinstance(job.job_type, JobType)
                        else job.job_type
                    )
                    if job_type_val != type_val:
                        continue

                if not include_completed and job.is_finished:
                    continue

                jobs.append(self.get_status(job.job_id))

                if len(jobs) >= limit:
                    break

            except Exception:
                continue  # Skip corrupted files

        return jobs

    # -------------------------------------------------------------------------
    # Job Control
    # -------------------------------------------------------------------------

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns
        -------
            True if cancelled, False if not found or already finished
        """
        job = self.get_job(job_id)
        if not job:
            return False

        if job.is_finished:
            return False

        # Cancel asyncio task if running in this process
        if job_id in self._running_tasks:
            self._running_tasks[job_id].cancel()
            return True

        # Kill process if running in background
        if job.pid:
            try:
                os.kill(job.pid, signal.SIGTERM)
                job.cancel()
                job.save(self.jobs_dir)
                return True
            except ProcessLookupError:
                # Process already dead
                job.cancel()
                job.save(self.jobs_dir)
                return True
            except Exception:
                return False

        # Mark as cancelled if pending
        if job.status == JobStatus.PENDING:
            job.cancel()
            job.save(self.jobs_dir)
            return True

        return False

    async def wait_for_job(
        self,
        job_id: str,
        timeout: float | None = None,
        poll_interval: float = 1.0,
    ) -> dict[str, Any] | None:
        """Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks

        Returns
        -------
            Job result or None if timeout/not found
        """
        start_time = datetime.now()

        while True:
            job = self.get_job(job_id)
            if not job:
                return None

            if job.is_finished:
                return job.to_dict()

            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    return None

            await asyncio.sleep(poll_interval)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def cleanup_old_jobs(
        self,
        max_age_hours: int = 24,
        keep_failed: bool = True,
    ) -> int:
        """Remove old completed jobs.

        Args:
            max_age_hours: Maximum age of jobs to keep
            keep_failed: Don't delete failed jobs

        Returns
        -------
            Number of jobs deleted
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        deleted = 0

        for job_file in self.jobs_dir.glob("*.json"):
            try:
                job = Job.load(job_file)

                if not job.is_finished:
                    continue

                if keep_failed and job.status == JobStatus.FAILED:
                    continue

                completed_at = datetime.fromisoformat(job.completed_at)
                if completed_at < cutoff:
                    job_file.unlink()
                    deleted += 1

            except Exception:
                continue

        return deleted

    def delete_job(self, job_id: str) -> bool:
        """Delete a job file.

        Args:
            job_id: Job ID to delete

        Returns
        -------
            True if deleted, False if not found
        """
        job_file = self.jobs_dir / f"{job_id}.json"
        if job_file.exists():
            job_file.unlink()
            self._running_jobs.pop(job_id, None)
            self._running_tasks.pop(job_id, None)
            return True
        return False


# Singleton instance
_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """Get the global JobManager instance."""
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


# EOF
