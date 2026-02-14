#!/usr/bin/env python3
# Timestamp: 2026-02-14
# File: scitex/events/_types.py

"""Known event types for the SciTeX event bus."""

from __future__ import annotations

from typing import Any

EVENT_TYPES: dict[str, dict[str, Any]] = {
    "test_complete": {
        "description": "Test suite completed (local or HPC)",
        "payload_keys": ["exit_code", "module", "log_tail"],
    },
    "job_done": {
        "description": "HPC/Slurm job completed",
        "payload_keys": ["job_id", "host", "state"],
    },
    "build_result": {
        "description": "LaTeX manuscript build completed",
        "payload_keys": ["doc_type", "success", "errors"],
    },
    "scholar_done": {
        "description": "Scholar paper fetch/enrichment completed",
        "payload_keys": ["count", "failed", "project"],
    },
    "stats_done": {
        "description": "Long-running statistical computation completed",
        "payload_keys": ["test_name", "p_value", "duration"],
    },
}


def list_types() -> list[str]:
    """Return list of known event type names."""
    return sorted(EVENT_TYPES.keys())


def get_type_info(event_type: str) -> dict[str, Any]:
    """Return metadata for an event type, or empty dict if unknown."""
    return EVENT_TYPES.get(event_type, {})


# EOF
