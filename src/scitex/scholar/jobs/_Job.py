#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/scholar/jobs/_Job.py
# ----------------------------------------

"""Job dataclass for async task management."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Supported job types."""

    FETCH = "fetch"
    FETCH_BIBTEX = "fetch_bibtex"
    ENRICH = "enrich"
    DOWNLOAD_PDF = "download_pdf"


@dataclass
class JobProgress:
    """Progress tracking for a job."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    current_item: str | None = None
    message: str | None = None

    @property
    def percent(self) -> float:
        """Calculate completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "current_item": self.current_item,
            "message": self.message,
            "percent": round(self.percent, 1),
        }


@dataclass
class Job:
    """Represents an async job."""

    # Core identifiers
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    job_type: JobType | str = JobType.FETCH

    # Parameters for the job
    params: dict[str, Any] = field(default_factory=dict)

    # Status tracking
    status: JobStatus = JobStatus.PENDING
    progress: JobProgress = field(default_factory=JobProgress)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str | None = None
    completed_at: str | None = None

    # Result data
    result: dict[str, Any] | None = None
    error: str | None = None

    # Structured error (enhanced error info with categorization and screenshots)
    structured_error: dict[str, Any] | None = None

    # Process tracking (for background execution)
    pid: int | None = None

    def start(self) -> None:
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now().isoformat()

    def complete(self, result: dict[str, Any]) -> None:
        """Mark job as completed with result."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        self.result = result
        self.pid = None

    def fail(
        self,
        error: str,
        structured_error: dict[str, Any] | None = None,
    ) -> None:
        """Mark job as failed with error message.

        Args:
            error: Simple error message string
            structured_error: Optional structured error dict with categorization,
                             screenshot paths, and suggested actions
        """
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now().isoformat()
        self.error = error
        self.structured_error = structured_error
        self.pid = None

    def cancel(self) -> None:
        """Mark job as cancelled."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now().isoformat()
        self.pid = None

    def update_progress(
        self,
        completed: int | None = None,
        total: int | None = None,
        failed: int | None = None,
        current_item: str | None = None,
        message: str | None = None,
    ) -> None:
        """Update job progress."""
        if completed is not None:
            self.progress.completed = completed
        if total is not None:
            self.progress.total = total
        if failed is not None:
            self.progress.failed = failed
        if current_item is not None:
            self.progress.current_item = current_item
        if message is not None:
            self.progress.message = message

    @property
    def is_finished(self) -> bool:
        """Check if job has finished (completed, failed, or cancelled)."""
        return self.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == JobStatus.RUNNING

    @property
    def duration_seconds(self) -> float | None:
        """Calculate job duration in seconds."""
        if not self.started_at:
            return None

        start = datetime.fromisoformat(self.started_at)

        if self.completed_at:
            end = datetime.fromisoformat(self.completed_at)
        else:
            end = datetime.now()

        return (end - start).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value
            if isinstance(self.job_type, JobType)
            else self.job_type,
            "params": self.params,
            "status": self.status.value,
            "progress": self.progress.to_dict(),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "structured_error": self.structured_error,
            "pid": self.pid,
            "duration_seconds": self.duration_seconds,
        }

    def to_json(self) -> str:
        """Serialize job to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        """Create Job instance from dictionary."""
        # Handle enums
        job_type = data.get("job_type", "fetch")
        if isinstance(job_type, str):
            try:
                job_type = JobType(job_type)
            except ValueError:
                pass  # Keep as string if not a valid JobType

        status = data.get("status", "pending")
        if isinstance(status, str):
            status = JobStatus(status)

        # Handle progress
        progress_data = data.get("progress", {})
        if isinstance(progress_data, dict):
            progress = JobProgress(
                total=progress_data.get("total", 0),
                completed=progress_data.get("completed", 0),
                failed=progress_data.get("failed", 0),
                current_item=progress_data.get("current_item"),
                message=progress_data.get("message"),
            )
        else:
            progress = JobProgress()

        return cls(
            job_id=data.get("job_id", uuid.uuid4().hex[:12]),
            job_type=job_type,
            params=data.get("params", {}),
            status=status,
            progress=progress,
            created_at=data.get("created_at", datetime.now().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error"),
            structured_error=data.get("structured_error"),
            pid=data.get("pid"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Job:
        """Create Job instance from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, jobs_dir: Path) -> Path:
        """Save job to file."""
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job_file = jobs_dir / f"{self.job_id}.json"
        job_file.write_text(self.to_json())
        return job_file

    @classmethod
    def load(cls, job_file: Path) -> Job:
        """Load job from file."""
        return cls.from_json(job_file.read_text())


# EOF
