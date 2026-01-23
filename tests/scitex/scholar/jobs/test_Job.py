#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: tests/scitex/scholar/jobs/test_Job.py
# ----------------------------------------

"""Tests for Job dataclass and related classes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scitex.scholar.jobs._Job import Job, JobProgress, JobStatus, JobType


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_status_is_string_enum(self):
        """Test JobStatus is a string enum."""
        assert isinstance(JobStatus.PENDING, str)
        assert JobStatus.PENDING == "pending"


class TestJobType:
    """Tests for JobType enum."""

    def test_type_values(self):
        """Test all job type values exist."""
        assert JobType.FETCH.value == "fetch"
        assert JobType.FETCH_BIBTEX.value == "fetch_bibtex"
        assert JobType.ENRICH.value == "enrich"
        assert JobType.DOWNLOAD_PDF.value == "download_pdf"


class TestJobProgress:
    """Tests for JobProgress dataclass."""

    def test_default_values(self):
        """Test default progress values."""
        progress = JobProgress()
        assert progress.total == 0
        assert progress.completed == 0
        assert progress.failed == 0
        assert progress.current_item is None
        assert progress.message is None

    def test_percent_with_zero_total(self):
        """Test percent calculation with zero total."""
        progress = JobProgress(total=0, completed=0)
        assert progress.percent == 0.0

    def test_percent_calculation(self):
        """Test percent calculation with values."""
        progress = JobProgress(total=10, completed=5)
        assert progress.percent == 50.0

    def test_percent_full_completion(self):
        """Test percent at 100%."""
        progress = JobProgress(total=10, completed=10)
        assert progress.percent == 100.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        progress = JobProgress(
            total=10,
            completed=5,
            failed=1,
            current_item="paper.pdf",
            message="Downloading",
        )
        result = progress.to_dict()
        assert result["total"] == 10
        assert result["completed"] == 5
        assert result["failed"] == 1
        assert result["current_item"] == "paper.pdf"
        assert result["message"] == "Downloading"
        assert result["percent"] == 50.0


class TestJob:
    """Tests for Job dataclass."""

    def test_default_job_creation(self):
        """Test creating job with defaults."""
        job = Job()
        assert len(job.job_id) == 12
        assert job.job_type == JobType.FETCH
        assert job.status == JobStatus.PENDING
        assert job.params == {}
        assert job.result is None
        assert job.error is None

    def test_job_with_params(self):
        """Test creating job with parameters."""
        job = Job(
            job_type=JobType.FETCH,
            params={"dois": ["10.1234/test"], "project": "test"},
        )
        assert job.params["dois"] == ["10.1234/test"]
        assert job.params["project"] == "test"

    def test_start_job(self):
        """Test starting a job."""
        job = Job()
        assert job.status == JobStatus.PENDING
        assert job.started_at is None

        job.start()
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None

    def test_complete_job(self):
        """Test completing a job."""
        job = Job()
        job.start()
        result = {"success": True, "papers": 5}
        job.complete(result)

        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.result == result
        assert job.pid is None

    def test_fail_job(self):
        """Test failing a job."""
        job = Job()
        job.start()
        job.fail("Connection timeout")

        assert job.status == JobStatus.FAILED
        assert job.completed_at is not None
        assert job.error == "Connection timeout"
        assert job.pid is None

    def test_cancel_job(self):
        """Test cancelling a job."""
        job = Job()
        job.start()
        job.pid = 12345
        job.cancel()

        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None
        assert job.pid is None

    def test_update_progress(self):
        """Test updating job progress."""
        job = Job()
        job.update_progress(total=10, completed=3, message="Processing")

        assert job.progress.total == 10
        assert job.progress.completed == 3
        assert job.progress.message == "Processing"

    def test_update_progress_partial(self):
        """Test partial progress update."""
        job = Job()
        job.update_progress(total=10)
        job.update_progress(completed=5)
        job.update_progress(current_item="file.pdf")

        assert job.progress.total == 10
        assert job.progress.completed == 5
        assert job.progress.current_item == "file.pdf"

    def test_is_finished_pending(self):
        """Test is_finished for pending job."""
        job = Job()
        assert not job.is_finished

    def test_is_finished_running(self):
        """Test is_finished for running job."""
        job = Job()
        job.start()
        assert not job.is_finished

    def test_is_finished_completed(self):
        """Test is_finished for completed job."""
        job = Job()
        job.complete({"success": True})
        assert job.is_finished

    def test_is_finished_failed(self):
        """Test is_finished for failed job."""
        job = Job()
        job.fail("Error")
        assert job.is_finished

    def test_is_finished_cancelled(self):
        """Test is_finished for cancelled job."""
        job = Job()
        job.cancel()
        assert job.is_finished

    def test_is_running(self):
        """Test is_running property."""
        job = Job()
        assert not job.is_running
        job.start()
        assert job.is_running
        job.complete({})
        assert not job.is_running

    def test_duration_not_started(self):
        """Test duration before job starts."""
        job = Job()
        assert job.duration_seconds is None

    def test_duration_running(self):
        """Test duration for running job."""
        job = Job()
        job.start()
        duration = job.duration_seconds
        assert duration is not None
        assert duration >= 0

    def test_duration_completed(self):
        """Test duration for completed job."""
        job = Job()
        job.start()
        job.complete({})
        duration = job.duration_seconds
        assert duration is not None
        assert duration >= 0


class TestJobSerialization:
    """Tests for Job serialization."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        job = Job(job_id="test123", job_type=JobType.FETCH)
        job.update_progress(total=5, completed=2)
        result = job.to_dict()

        assert result["job_id"] == "test123"
        assert result["job_type"] == "fetch"
        assert result["status"] == "pending"
        assert result["progress"]["total"] == 5
        assert result["progress"]["completed"] == 2

    def test_to_json(self):
        """Test JSON serialization."""
        job = Job(job_id="test123")
        json_str = job.to_json()
        data = json.loads(json_str)

        assert data["job_id"] == "test123"
        assert isinstance(json_str, str)

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "job_id": "abc123",
            "job_type": "fetch",
            "status": "running",
            "params": {"doi": "10.1234/test"},
            "progress": {"total": 10, "completed": 5},
        }
        job = Job.from_dict(data)

        assert job.job_id == "abc123"
        assert job.job_type == JobType.FETCH
        assert job.status == JobStatus.RUNNING
        assert job.params["doi"] == "10.1234/test"
        assert job.progress.total == 10

    def test_from_json(self):
        """Test creation from JSON string."""
        data = {"job_id": "xyz789", "job_type": "enrich", "status": "completed"}
        json_str = json.dumps(data)
        job = Job.from_json(json_str)

        assert job.job_id == "xyz789"
        assert job.job_type == JobType.ENRICH
        assert job.status == JobStatus.COMPLETED

    def test_roundtrip_serialization(self):
        """Test serialization roundtrip."""
        original = Job(
            job_id="roundtrip",
            job_type=JobType.DOWNLOAD_PDF,
            params={"url": "http://example.com"},
        )
        original.start()
        original.update_progress(total=5, completed=3)

        json_str = original.to_json()
        restored = Job.from_json(json_str)

        assert restored.job_id == original.job_id
        assert restored.job_type == original.job_type
        assert restored.status == original.status
        assert restored.progress.total == original.progress.total


class TestJobPersistence:
    """Tests for Job file persistence."""

    def test_save_and_load(self):
        """Test saving and loading job from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            job = Job(job_id="persist_test", params={"key": "value"})
            job.start()

            job_file = job.save(jobs_dir)
            assert job_file.exists()
            assert job_file.name == "persist_test.json"

            loaded = Job.load(job_file)
            assert loaded.job_id == job.job_id
            assert loaded.params == job.params
            assert loaded.status == job.status

    def test_save_creates_directory(self):
        """Test save creates directory if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir) / "nested" / "jobs"
            job = Job()
            job.save(jobs_dir)

            assert jobs_dir.exists()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
