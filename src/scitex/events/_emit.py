#!/usr/bin/env python3
# Timestamp: 2026-02-14
# File: scitex/events/_emit.py

"""Event emission: state files and optional webhook delivery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ._schema import Event

_MAX_HISTORY_LINES = 1000
_DEFAULT_API_URL = "https://scitex.ai/api/events/"


def emit(
    event_type: str,
    project: str,
    status: str = "success",
    payload: dict[str, Any] | None = None,
    source: str = "local",
) -> Event:
    """Emit an event to state file and optional webhook.

    Always writes to ~/.scitex/events/{type}_latest.json.
    Also appends to ~/.scitex/events/history.jsonl.
    If SCITEX_API_KEY is set, POSTs to the cloud API (best-effort).

    Parameters
    ----------
    event_type : str
        Event type (e.g., "test_complete").
    project : str
        Project name.
    status : str
        "success" or "failure".
    payload : dict, optional
        Arbitrary event data.
    source : str
        Origin: "local", "hpc", "ci".

    Returns
    -------
    Event
        The emitted event object.
    """
    event = Event(
        type=event_type,
        project=project,
        status=status,
        payload=payload or {},
        source=source,
    )

    _write_state_file(event)
    _append_history(event)
    _post_webhook(event)

    return event


def latest(event_type: str | None = None) -> dict[str, Any] | None:
    """Read the latest event from state files.

    Parameters
    ----------
    event_type : str, optional
        If given, read the latest event of this type.
        If None, read the most recent event across all types.

    Returns
    -------
    dict or None
        Event data, or None if no events found.
    """
    events_dir = _events_dir()
    if not events_dir.exists():
        return None

    if event_type:
        state_file = events_dir / f"{event_type}_latest.json"
        if state_file.exists():
            return json.loads(state_file.read_text(encoding="utf-8"))
        return None

    # Find most recent across all types
    latest_files = sorted(
        events_dir.glob("*_latest.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if latest_files:
        return json.loads(latest_files[0].read_text(encoding="utf-8"))
    return None


def history(limit: int = 20) -> list:
    """Read recent events from history file.

    Parameters
    ----------
    limit : int
        Maximum number of events to return.

    Returns
    -------
    list of dict
        Events in reverse chronological order.
    """
    history_file = _events_dir() / "history.jsonl"
    if not history_file.exists():
        return []

    lines = history_file.read_text(encoding="utf-8").strip().splitlines()
    events = []
    for line in reversed(lines[-limit:]):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


# -- Internal helpers --


def _events_dir() -> Path:
    """Get or create ~/.scitex/events/ directory."""
    d = Path.home() / ".scitex" / "events"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_state_file(event: Event) -> None:
    """Write per-type latest state file."""
    events_dir = _events_dir()
    state_file = events_dir / f"{event.type}_latest.json"
    state_file.write_text(
        json.dumps(event.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


def _append_history(event: Event) -> None:
    """Append to history.jsonl with rotation."""
    history_file = _events_dir() / "history.jsonl"

    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_dict()) + "\n")

    # Rotate if too large
    try:
        lines = history_file.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > _MAX_HISTORY_LINES:
            trimmed = lines[-_MAX_HISTORY_LINES:]
            history_file.write_text("\n".join(trimmed) + "\n", encoding="utf-8")
    except OSError:
        pass


def _post_webhook(event: Event) -> None:
    """POST event to cloud API if SCITEX_API_KEY is set. Best-effort."""
    api_key = os.environ.get("SCITEX_API_KEY", "")
    if not api_key:
        return

    api_url = os.environ.get("SCITEX_API_URL", _DEFAULT_API_URL)

    try:
        import urllib.request

        req = urllib.request.Request(
            api_url,
            data=json.dumps(event.to_dict()).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # Best effort — never fail the caller


# EOF
