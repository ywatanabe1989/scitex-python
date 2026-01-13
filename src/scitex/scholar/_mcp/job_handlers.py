#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/scholar/_mcp/job_handlers.py
# ----------------------------------------

"""Job management handlers for the scitex-scholar MCP server.

These handlers provide non-blocking fetch operations by delegating to the
CLI with --async flag and exposing job management functionality.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime

__all__ = [
    "fetch_papers_handler",
    "list_jobs_handler",
    "get_job_status_handler",
    "start_job_handler",
    "cancel_job_handler",
    "get_job_result_handler",
]


def _run_cli_json(*args: str) -> dict:
    """Run CLI command with --json and return parsed result."""
    cmd = [sys.executable, "-m", "scitex", "scholar", *args, "--json"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.stdout:
            return json.loads(result.stdout)
        elif result.stderr:
            return {"success": False, "error": result.stderr}
        else:
            return {"success": False, "error": "No output from CLI"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "CLI command timed out"}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON from CLI: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def fetch_papers_handler(
    papers: list[str] | None = None,
    bibtex_path: str | None = None,
    project: str | None = None,
    workers: int | None = None,
    browser_mode: str = "stealth",
    chrome_profile: str = "system",
    force: bool = False,
    output: str | None = None,
    async_mode: bool = True,
) -> dict:
    """Fetch papers to your library.

    This is the unified fetch handler that supports both synchronous and
    asynchronous (background job) modes. When async_mode=True (default),
    the fetch operation runs in the background and returns immediately
    with a job_id that can be used to track progress.

    Args:
        papers: List of DOIs or titles to fetch
        bibtex_path: Path to BibTeX file (alternative to papers)
        project: Project name for organizing papers
        workers: Number of parallel workers
        browser_mode: Browser mode ("stealth" or "interactive")
        chrome_profile: Chrome profile name
        force: Force re-download even if files exist
        output: Output path for enriched BibTeX (only with bibtex_path)
        async_mode: If True, run in background and return job_id immediately
    """
    # Build CLI arguments
    args = ["fetch"]

    if papers:
        args.extend(papers)
    if bibtex_path:
        args.extend(["--from-bibtex", bibtex_path])
    if project:
        args.extend(["--project", project])
    if workers:
        args.extend(["--workers", str(workers)])
    if browser_mode:
        args.extend(["--browser-mode", browser_mode])
    if chrome_profile:
        args.extend(["--chrome-profile", chrome_profile])
    if force:
        args.append("--force")
    if output:
        args.extend(["--output", output])
    if async_mode:
        args.append("--async")

    result = _run_cli_json(*args)
    result["timestamp"] = datetime.now().isoformat()
    return result


async def list_jobs_handler(
    status: str | None = None,
    limit: int = 20,
) -> dict:
    """List all background jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)
        limit: Maximum number of jobs to return
    """
    args = ["jobs", "list", "-n", str(limit)]
    if status:
        args.extend(["--status", status])

    result = _run_cli_json(*args)
    result["timestamp"] = datetime.now().isoformat()
    return result


async def get_job_status_handler(job_id: str) -> dict:
    """Get status of a specific job.

    Args:
        job_id: The job ID to check
    """
    result = _run_cli_json("jobs", "status", job_id)
    result["timestamp"] = datetime.now().isoformat()
    return result


async def start_job_handler(job_id: str) -> dict:
    """Start a pending job.

    Args:
        job_id: The job ID to start
    """
    result = _run_cli_json("jobs", "start", job_id)
    result["timestamp"] = datetime.now().isoformat()
    return result


async def cancel_job_handler(job_id: str) -> dict:
    """Cancel a running or pending job.

    Args:
        job_id: The job ID to cancel
    """
    result = _run_cli_json("jobs", "cancel", job_id)
    result["timestamp"] = datetime.now().isoformat()
    return result


async def get_job_result_handler(job_id: str) -> dict:
    """Get the result of a completed job.

    Args:
        job_id: The job ID to get results for
    """
    result = _run_cli_json("jobs", "result", job_id)
    result["timestamp"] = datetime.now().isoformat()
    return result


# EOF
