#!/usr/bin/env python3
# Timestamp: 2026-02-14
# File: scitex/events/_schema.py

"""Event schema for the SciTeX event bus."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Event:
    """A single event in the SciTeX event bus.

    Parameters
    ----------
    type : str
        Event type (e.g., "test_complete", "job_done", "build_result").
    project : str
        Project name this event belongs to.
    status : str
        Outcome: "success" or "failure".
    payload : dict
        Arbitrary data specific to the event type.
    source : str
        Where the event originated: "local", "hpc", "ci".
    timestamp : str
        ISO-format timestamp, auto-generated if not provided.
    """

    type: str
    project: str
    status: str = "success"
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "local"
    timestamp: str = ""

    def __post_init__(self):
        """Validate and set defaults."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Create Event from dictionary."""
        return cls(
            type=data["type"],
            project=data.get("project", ""),
            status=data.get("status", "unknown"),
            payload=data.get("payload", {}),
            source=data.get("source", "local"),
            timestamp=data.get("timestamp", ""),
        )


# EOF
