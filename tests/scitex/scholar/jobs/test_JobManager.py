#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: tests/scitex/scholar/jobs/test_JobManager.py
# ----------------------------------------

"""Tests for JobManager class."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scitex.scholar.jobs._Job import Job, JobStatus, JobType
from scitex.scholar.jobs._JobManager import JobManager


class TestJobManagerInit:
    """Tests for JobManager initialization."""

    def test_default_init(self):
        """Test initialization with default directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            assert manager.jobs_dir.exists()

    def test_creates_directory(self):
        """Test that init creates the jobs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir) / "nested" / "jobs"
            manager = JobManager(jobs_dir=jobs_dir)
            assert jobs_dir.exists()


class TestJobSubmission:
    """Tests for job submission."""

    def test_submit_returns_job_id(self):
        """Test submit returns a job ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(
                job_type=JobType.FETCH,
                params={"dois": ["10.1234/test"]},
            )
            assert len(job_id) == 12

    def test_submit_creates_file(self):
        """Test submit creates job file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(
                job_type=JobType.FETCH,
                params={},
            )
            job_file = Path(tmpdir) / f"{job_id}.json"
            assert job_file.exists()

    def test_submit_with_custom_id(self):
        """Test submit with custom job ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(
                job_type=JobType.FETCH,
                params={},
                job_id="custom123",
            )
            assert job_id == "custom123"

    def test_submitted_job_is_pending(self):
        """Test submitted job starts as pending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            job = manager.get_job(job_id)
            assert job.status == JobStatus.PENDING


class TestJobRetrieval:
    """Tests for job retrieval."""

    def test_get_job_existing(self):
        """Test getting an existing job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(
                job_type=JobType.FETCH,
                params={"key": "value"},
            )
            job = manager.get_job(job_id)
            assert job is not None
            assert job.job_id == job_id
            assert job.params["key"] == "value"

    def test_get_job_nonexistent(self):
        """Test getting a non-existent job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job = manager.get_job("nonexistent")
            assert job is None

    def test_get_status_existing(self):
        """Test getting status of existing job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            status = manager.get_status(job_id)
            assert status is not None
            assert status["job_id"] == job_id
            assert status["status"] == "pending"
            assert "progress" in status

    def test_get_status_nonexistent(self):
        """Test getting status of non-existent job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            status = manager.get_status("nonexistent")
            assert status is None

    def test_get_result_completed_job(self):
        """Test getting result from completed job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            job = manager.get_job(job_id)
            job.complete({"success": True, "papers": 5})
            job.save(manager.jobs_dir)
            result = manager.get_result(job_id)
            assert result == {"success": True, "papers": 5}

    def test_get_result_pending_job(self):
        """Test getting result from pending job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            result = manager.get_result(job_id)
            assert result is None


class TestListJobs:
    """Tests for listing jobs."""

    def test_list_jobs_empty(self):
        """Test listing with no jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            jobs = manager.list_jobs()
            assert jobs == []

    def test_list_jobs_returns_all(self):
        """Test listing returns all jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            manager.submit(job_type=JobType.FETCH, params={})
            manager.submit(job_type=JobType.ENRICH, params={})
            jobs = manager.list_jobs()
            assert len(jobs) == 2

    def test_list_jobs_filter_by_status(self):
        """Test filtering jobs by status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id1 = manager.submit(job_type=JobType.FETCH, params={})
            job_id2 = manager.submit(job_type=JobType.FETCH, params={})
            # Complete one job
            job = manager.get_job(job_id1)
            job.complete({})
            job.save(manager.jobs_dir)
            pending_jobs = manager.list_jobs(status=JobStatus.PENDING)
            assert len(pending_jobs) == 1
            assert pending_jobs[0]["job_id"] == job_id2

    def test_list_jobs_filter_by_type(self):
        """Test filtering jobs by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            manager.submit(job_type=JobType.FETCH, params={})
            manager.submit(job_type=JobType.ENRICH, params={})
            fetch_jobs = manager.list_jobs(job_type=JobType.FETCH)
            assert len(fetch_jobs) == 1
            assert fetch_jobs[0]["job_type"] == "fetch"

    def test_list_jobs_with_limit(self):
        """Test listing with limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            for _ in range(5):
                manager.submit(job_type=JobType.FETCH, params={})
            jobs = manager.list_jobs(limit=3)
            assert len(jobs) == 3

    def test_list_jobs_exclude_completed(self):
        """Test excluding completed jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id1 = manager.submit(job_type=JobType.FETCH, params={})
            manager.submit(job_type=JobType.FETCH, params={})
            job = manager.get_job(job_id1)
            job.complete({})
            job.save(manager.jobs_dir)
            jobs = manager.list_jobs(include_completed=False)
            assert len(jobs) == 1


class TestJobCancel:
    """Tests for job cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self):
        """Test cancelling a pending job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            result = await manager.cancel(job_id)
            assert result is True
            job = manager.get_job(job_id)
            assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self):
        """Test cancelling a non-existent job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            result = await manager.cancel("nonexistent")
            assert result is False

    @pytest.mark.asyncio
    async def test_cancel_completed_job(self):
        """Test cancelling an already completed job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            job = manager.get_job(job_id)
            job.complete({})
            job.save(manager.jobs_dir)
            result = await manager.cancel(job_id)
            assert result is False


class TestJobCleanup:
    """Tests for job cleanup."""

    def test_cleanup_old_jobs(self):
        """Test cleaning up old completed jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            job = manager.get_job(job_id)
            job.complete({})
            old_time = (datetime.now() - timedelta(hours=48)).isoformat()
            job.completed_at = old_time
            job.save(manager.jobs_dir)
            deleted = manager.cleanup_old_jobs(max_age_hours=24)
            assert deleted == 1
            assert manager.get_job(job_id) is None

    def test_cleanup_keeps_recent_jobs(self):
        """Test cleanup keeps recent jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            job = manager.get_job(job_id)
            job.complete({})
            job.save(manager.jobs_dir)
            deleted = manager.cleanup_old_jobs(max_age_hours=24)
            assert deleted == 0
            assert manager.get_job(job_id) is not None

    def test_cleanup_keeps_failed_jobs(self):
        """Test cleanup keeps failed jobs when requested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            job = manager.get_job(job_id)
            job.fail("Error")
            old_time = (datetime.now() - timedelta(hours=48)).isoformat()
            job.completed_at = old_time
            job.save(manager.jobs_dir)
            deleted = manager.cleanup_old_jobs(max_age_hours=24, keep_failed=True)
            assert deleted == 0

    def test_delete_job(self):
        """Test deleting a specific job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            assert manager.get_job(job_id) is not None
            result = manager.delete_job(job_id)
            assert result is True
            assert manager.get_job(job_id) is None

    def test_delete_nonexistent_job(self):
        """Test deleting non-existent job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            result = manager.delete_job("nonexistent")
            assert result is False


class TestAsyncExecution:
    """Tests for async job execution."""

    @pytest.mark.asyncio
    async def test_submit_async_with_executor(self):
        """Test submitting job with async executor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))

            async def mock_executor(progress_callback=None, **kwargs):
                if progress_callback:
                    progress_callback(total=10, completed=10)
                return {"success": True}

            job_id = await manager.submit_async(
                job_type=JobType.FETCH,
                params={},
                executor=mock_executor,
            )
            await asyncio.sleep(0.1)
            job = manager.get_job(job_id)
            assert job.status == JobStatus.COMPLETED
            assert job.result == {"success": True}

    @pytest.mark.asyncio
    async def test_submit_async_with_failing_executor(self):
        """Test job failure during execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))

            async def failing_executor(**kwargs):
                raise ValueError("Test error")

            job_id = await manager.submit_async(
                job_type=JobType.FETCH,
                params={},
                executor=failing_executor,
            )
            await asyncio.sleep(0.1)
            job = manager.get_job(job_id)
            assert job.status == JobStatus.FAILED
            assert "Test error" in job.error

    @pytest.mark.asyncio
    async def test_wait_for_job(self):
        """Test waiting for job completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))

            async def quick_executor(**kwargs):
                return {"done": True}

            job_id = await manager.submit_async(
                job_type=JobType.FETCH,
                params={},
                executor=quick_executor,
            )
            result = await manager.wait_for_job(job_id, timeout=5.0)
            assert result is not None
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_wait_for_job_timeout(self):
        """Test wait timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(jobs_dir=Path(tmpdir))
            job_id = manager.submit(job_type=JobType.FETCH, params={})
            result = await manager.wait_for_job(job_id, timeout=0.1, poll_interval=0.05)
            assert result is None


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
