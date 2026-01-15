#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: tests/scitex/scholar/_mcp/test_job_handlers.py
# ----------------------------------------

"""Tests for MCP job handlers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar._mcp.job_handlers import (
    _run_cli_json,
    cancel_job_handler,
    fetch_papers_handler,
    get_job_result_handler,
    get_job_status_handler,
    list_jobs_handler,
    start_job_handler,
)


class TestRunCliJson:
    """Tests for _run_cli_json helper."""

    def test_returns_parsed_json(self):
        """Test successful JSON parsing."""
        mock_result = MagicMock()
        mock_result.stdout = '{"success": true, "data": "test"}'
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _run_cli_json("test", "args")
            assert result["success"] is True
            assert result["data"] == "test"

    def test_returns_error_on_stderr(self):
        """Test error handling from stderr."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "Error message"
        with patch("subprocess.run", return_value=mock_result):
            result = _run_cli_json("test")
            assert result["success"] is False
            assert "Error message" in result["error"]

    def test_handles_timeout(self):
        """Test timeout handling."""
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300)):
            result = _run_cli_json("test")
            assert result["success"] is False
            assert "timed out" in result["error"]

    def test_handles_invalid_json(self):
        """Test handling of invalid JSON."""
        mock_result = MagicMock()
        mock_result.stdout = "not valid json"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _run_cli_json("test")
            assert result["success"] is False
            assert "Invalid JSON" in result["error"]


class TestFetchPapersHandler:
    """Tests for fetch_papers_handler."""

    @pytest.mark.asyncio
    async def test_fetch_with_papers(self):
        """Test fetch with DOI list."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"success": True, "job_id": "abc123"})
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = await fetch_papers_handler(
                papers=["10.1234/test"],
                project="test-project",
                async_mode=True,
            )
            assert result["success"] is True
            assert result["job_id"] == "abc123"
            call_args = mock_run.call_args[0][0]
            assert "fetch" in call_args
            assert "10.1234/test" in call_args
            assert "--project" in call_args
            assert "--async" in call_args

    @pytest.mark.asyncio
    async def test_fetch_with_bibtex(self):
        """Test fetch with bibtex file."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"success": True})
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            await fetch_papers_handler(
                bibtex_path="/path/to/papers.bib",
                async_mode=False,
            )
            call_args = mock_run.call_args[0][0]
            assert "--from-bibtex" in call_args
            assert "/path/to/papers.bib" in call_args
            assert "--async" not in call_args

    @pytest.mark.asyncio
    async def test_fetch_adds_timestamp(self):
        """Test that timestamp is added to result."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"success": True})
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = await fetch_papers_handler()
            assert "timestamp" in result


class TestListJobsHandler:
    """Tests for list_jobs_handler."""

    @pytest.mark.asyncio
    async def test_list_jobs_default(self):
        """Test listing jobs with defaults."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"success": True, "jobs": []})
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = await list_jobs_handler()
            assert result["success"] is True
            call_args = mock_run.call_args[0][0]
            assert "jobs" in call_args
            assert "list" in call_args
            assert "-n" in call_args
            assert "20" in call_args

    @pytest.mark.asyncio
    async def test_list_jobs_with_filters(self):
        """Test listing jobs with status filter."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"success": True, "jobs": []})
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            await list_jobs_handler(status="running", limit=10)
            call_args = mock_run.call_args[0][0]
            assert "--status" in call_args
            assert "running" in call_args
            assert "10" in call_args


class TestGetJobStatusHandler:
    """Tests for get_job_status_handler."""

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting job status."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "success": True,
                "job_id": "abc123",
                "status": "running",
            }
        )
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = await get_job_status_handler("abc123")
            assert result["success"] is True
            assert result["job_id"] == "abc123"
            call_args = mock_run.call_args[0][0]
            assert "jobs" in call_args
            assert "status" in call_args
            assert "abc123" in call_args


class TestStartJobHandler:
    """Tests for start_job_handler."""

    @pytest.mark.asyncio
    async def test_start_job(self):
        """Test starting a job."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"success": True, "message": "Started"})
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = await start_job_handler("job123")
            assert result["success"] is True
            call_args = mock_run.call_args[0][0]
            assert "jobs" in call_args
            assert "start" in call_args
            assert "job123" in call_args


class TestCancelJobHandler:
    """Tests for cancel_job_handler."""

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        """Test cancelling a job."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"success": True, "message": "Cancelled"})
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = await cancel_job_handler("job123")
            assert result["success"] is True
            call_args = mock_run.call_args[0][0]
            assert "jobs" in call_args
            assert "cancel" in call_args
            assert "job123" in call_args


class TestGetJobResultHandler:
    """Tests for get_job_result_handler."""

    @pytest.mark.asyncio
    async def test_get_result(self):
        """Test getting job result."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "success": True,
                "result": {"papers": 5, "downloaded": 3},
            }
        )
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = await get_job_result_handler("job123")
            assert result["success"] is True
            call_args = mock_run.call_args[0][0]
            assert "jobs" in call_args
            assert "result" in call_args
            assert "job123" in call_args


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
