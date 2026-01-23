#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/scholar/_mcp/job_tool_schemas.py
# ----------------------------------------

"""Job management tool schemas for the scitex-scholar MCP server."""

from __future__ import annotations

import mcp.types as types

__all__ = ["get_job_tool_schemas"]


def get_job_tool_schemas() -> list[types.Tool]:
    """Return job management tool schemas."""
    return [
        # Unified fetch with async support
        types.Tool(
            name="fetch_papers",
            description=(
                "Fetch papers to your library. Supports async mode (default) which "
                "returns immediately with a job_id for tracking. Use list_jobs and "
                "get_job_status to monitor progress."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of DOIs or titles to fetch",
                    },
                    "bibtex_path": {
                        "type": "string",
                        "description": "Path to BibTeX file (alternative to papers)",
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name for organizing papers",
                    },
                    "workers": {
                        "type": "integer",
                        "description": "Number of parallel workers",
                    },
                    "browser_mode": {
                        "type": "string",
                        "description": "Browser mode for PDF download",
                        "enum": ["stealth", "interactive"],
                        "default": "stealth",
                    },
                    "chrome_profile": {
                        "type": "string",
                        "description": "Chrome profile name",
                        "default": "system",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force re-download even if files exist",
                        "default": False,
                    },
                    "output": {
                        "type": "string",
                        "description": "Output path for enriched BibTeX",
                    },
                    "async_mode": {
                        "type": "boolean",
                        "description": (
                            "If True, run in background and return job_id immediately. "
                            "If False, block until completion."
                        ),
                        "default": True,
                    },
                },
            },
        ),
        # Job listing
        types.Tool(
            name="list_jobs",
            description="List all background jobs with their status",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status",
                        "enum": [
                            "pending",
                            "running",
                            "completed",
                            "failed",
                            "cancelled",
                        ],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of jobs to return",
                        "default": 20,
                    },
                },
            },
        ),
        # Job status
        types.Tool(
            name="get_job_status",
            description="Get detailed status of a specific job including progress",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to check",
                    },
                },
                "required": ["job_id"],
            },
        ),
        # Start job
        types.Tool(
            name="start_job",
            description="Start a pending job that was submitted with async mode",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to start",
                    },
                },
                "required": ["job_id"],
            },
        ),
        # Cancel job
        types.Tool(
            name="cancel_job",
            description="Cancel a running or pending job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to cancel",
                    },
                },
                "required": ["job_id"],
            },
        ),
        # Get job result
        types.Tool(
            name="get_job_result",
            description="Get the result of a completed job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to get results for",
                    },
                },
                "required": ["job_id"],
            },
        ),
    ]


# EOF
