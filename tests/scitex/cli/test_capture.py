#!/usr/bin/env python3
"""Tests for scitex.cli.capture - Screenshot capture CLI commands."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.capture import capture


class TestCaptureGroup:
    """Tests for the capture command group."""

    def test_capture_help(self):
        """Test that capture help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(capture, ["--help"])
        assert result.exit_code == 0
        assert "Screen capture and monitoring" in result.output

    def test_capture_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(capture, ["--help"])
        expected_commands = ["snap", "start", "stop", "gif", "info", "window"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in capture help"


class TestCaptureSnap:
    """Tests for the capture snap command."""

    def test_snap_default(self):
        """Test snap command with default options."""
        runner = CliRunner()
        with patch("scitex.capture.snap") as mock_snap:
            mock_snap.return_value = "/tmp/screenshot.jpg"
            result = runner.invoke(capture, ["snap"])
            assert result.exit_code == 0
            assert "Screenshot saved" in result.output
            mock_snap.assert_called_once()

    def test_snap_with_output(self):
        """Test snap command with output path."""
        runner = CliRunner()
        with patch("scitex.capture.snap") as mock_snap:
            mock_snap.return_value = "/tmp/custom.jpg"
            result = runner.invoke(capture, ["snap", "--output", "/tmp/custom.jpg"])
            assert result.exit_code == 0
            call_kwargs = mock_snap.call_args[1]
            # CLI maps --output to output_dir parameter
            assert call_kwargs.get("output_dir") == "/tmp/custom.jpg"

    def test_snap_with_monitor(self):
        """Test snap command with specific monitor."""
        runner = CliRunner()
        with patch("scitex.capture.snap") as mock_snap:
            mock_snap.return_value = "/tmp/screenshot.jpg"
            result = runner.invoke(capture, ["snap", "--monitor", "1"])
            assert result.exit_code == 0
            call_kwargs = mock_snap.call_args[1]
            assert call_kwargs.get("monitor_id") == 1

    def test_snap_all_monitors(self):
        """Test snap command with --all-monitors flag."""
        runner = CliRunner()
        with patch("scitex.capture.snap") as mock_snap:
            mock_snap.return_value = "/tmp/screenshot.jpg"
            result = runner.invoke(capture, ["snap", "--all-monitors"])
            assert result.exit_code == 0
            call_kwargs = mock_snap.call_args[1]
            assert call_kwargs.get("capture_all") is True

    def test_snap_with_quality(self):
        """Test snap command with quality option."""
        runner = CliRunner()
        with patch("scitex.capture.snap") as mock_snap:
            mock_snap.return_value = "/tmp/screenshot.jpg"
            result = runner.invoke(capture, ["snap", "--quality", "90"])
            assert result.exit_code == 0
            call_kwargs = mock_snap.call_args[1]
            assert call_kwargs.get("quality") == 90

    def test_snap_error_handling(self):
        """Test snap command handles errors gracefully."""
        runner = CliRunner()
        with patch("scitex.capture.snap") as mock_snap:
            mock_snap.side_effect = Exception("Screenshot failed")
            result = runner.invoke(capture, ["snap"])
            assert result.exit_code == 1
            assert "Error" in result.output


class TestCaptureStart:
    """Tests for the capture start command."""

    def test_start_default(self):
        """Test start command with default options."""
        runner = CliRunner()
        with patch("scitex.capture.start") as mock_start:
            mock_start.return_value = {"session_id": "20250108_120000"}
            result = runner.invoke(capture, ["start"])
            assert result.exit_code == 0
            assert "Monitoring started" in result.output
            mock_start.assert_called_once()

    def test_start_with_interval(self):
        """Test start command with custom interval."""
        runner = CliRunner()
        with patch("scitex.capture.start") as mock_start:
            mock_start.return_value = {"session_id": "20250108_120000"}
            result = runner.invoke(capture, ["start", "--interval", "2.0"])
            assert result.exit_code == 0
            call_kwargs = mock_start.call_args[1]
            assert call_kwargs.get("interval") == 2.0

    def test_start_with_output_dir(self):
        """Test start command with output directory."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.capture.start") as mock_start:
                mock_start.return_value = {"session_id": "20250108_120000"}
                result = runner.invoke(capture, ["start", "--output", tmpdir])
                assert result.exit_code == 0


class TestCaptureStop:
    """Tests for the capture stop command."""

    def test_stop(self):
        """Test stop command."""
        runner = CliRunner()
        with patch("scitex.capture.stop") as mock_stop:
            mock_stop.return_value = {"stopped": True, "count": 10}
            result = runner.invoke(capture, ["stop"])
            assert result.exit_code == 0
            assert "Monitoring stopped" in result.output

    def test_stop_not_running(self):
        """Test stop command when not running."""
        runner = CliRunner()
        with patch("scitex.capture.stop") as mock_stop:
            mock_stop.return_value = {"stopped": False}
            result = runner.invoke(capture, ["stop"])
            assert result.exit_code == 0


class TestCaptureGif:
    """Tests for the capture gif command."""

    def test_gif_from_session(self):
        """Test gif command with session ID."""
        runner = CliRunner()
        with patch("scitex.capture.create_gif_from_session") as mock_gif:
            mock_gif.return_value = "/tmp/output.gif"
            result = runner.invoke(capture, ["gif", "--session", "20250108_120000"])
            assert result.exit_code == 0
            assert "GIF created" in result.output

    def test_gif_from_latest(self):
        """Test gif command with latest session."""
        runner = CliRunner()
        with patch("scitex.capture.create_gif_from_latest_session") as mock_gif:
            mock_gif.return_value = "/tmp/output.gif"
            result = runner.invoke(capture, ["gif"])
            assert result.exit_code == 0
            assert "GIF created" in result.output

    def test_gif_with_output(self):
        """Test gif command with output path."""
        runner = CliRunner()
        with patch("scitex.capture.create_gif_from_session") as mock_gif:
            mock_gif.return_value = "/tmp/custom.gif"
            result = runner.invoke(
                capture,
                ["gif", "--session", "20250108_120000", "--output", "/tmp/custom.gif"],
            )
            assert result.exit_code == 0
            call_kwargs = mock_gif.call_args[1]
            assert call_kwargs.get("output_path") == "/tmp/custom.gif"

    def test_gif_with_duration(self):
        """Test gif command with frame duration."""
        runner = CliRunner()
        with patch("scitex.capture.create_gif_from_session") as mock_gif:
            mock_gif.return_value = "/tmp/output.gif"
            result = runner.invoke(
                capture, ["gif", "--session", "20250108_120000", "--duration", "0.3"]
            )
            assert result.exit_code == 0
            call_kwargs = mock_gif.call_args[1]
            assert call_kwargs.get("duration") == 0.3

    def test_gif_failure(self):
        """Test gif command handles failure."""
        runner = CliRunner()
        with patch("scitex.capture.create_gif_from_latest_session") as mock_gif:
            mock_gif.return_value = None
            result = runner.invoke(capture, ["gif"])
            assert result.exit_code == 0  # Still 0 but with warning message
            assert "No screenshots found" in result.output


class TestCaptureInfo:
    """Tests for the capture info command."""

    def test_info_default(self):
        """Test info command."""
        runner = CliRunner()
        with patch("scitex.capture.get_info") as mock_info:
            mock_info.return_value = {
                "Monitors": {
                    "Count": 1,
                    "Details": [{"Width": 1920, "Height": 1080, "Primary": True}],
                },
                "Windows": {"Count": 0, "Details": []},
            }
            result = runner.invoke(capture, ["info"])
            assert result.exit_code == 0
            assert "Display Information" in result.output
            assert "Monitors" in result.output

    def test_info_json(self):
        """Test info command with --json flag."""
        runner = CliRunner()
        with patch("scitex.capture.get_info") as mock_info:
            mock_info.return_value = {
                "Monitors": {"Count": 1, "Details": [{"Width": 1920, "Height": 1080}]},
            }
            result = runner.invoke(capture, ["info", "--json"])
            assert result.exit_code == 0
            output = json.loads(result.output)
            assert "Monitors" in output


class TestCaptureWindow:
    """Tests for the capture window command."""

    def test_window_capture(self):
        """Test window capture command."""
        runner = CliRunner()
        with patch("scitex.capture.capture_window") as mock_capture:
            mock_capture.return_value = "/tmp/window.jpg"
            result = runner.invoke(capture, ["window", "12345"])
            assert result.exit_code == 0
            assert "Window captured" in result.output
            mock_capture.assert_called_once()

    def test_window_with_output(self):
        """Test window capture with output path."""
        runner = CliRunner()
        with patch("scitex.capture.capture_window") as mock_capture:
            mock_capture.return_value = "/tmp/custom.jpg"
            result = runner.invoke(
                capture, ["window", "12345", "--output", "/tmp/custom.jpg"]
            )
            assert result.exit_code == 0

    def test_window_failure(self):
        """Test window capture handles failure."""
        runner = CliRunner()
        with patch("scitex.capture.capture_window") as mock_capture:
            mock_capture.return_value = None
            result = runner.invoke(capture, ["window", "12345"])
            assert result.exit_code == 0  # Still 0 since window was captured
            assert "Window captured" in result.output


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
