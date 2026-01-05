#!/usr/bin/env python3
"""Tests for scitex.capture.cli module.

Tests CLI functionality:
- Argument parsing
- --list, --info, --stop, --gif actions
- Default capture behavior
- Error handling
"""

import os
import sys
import tempfile
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_main_function_exists(self):
        """Test main function exists and is callable."""
        from scitex.capture.cli import main

        assert callable(main)

    def test_help_argument(self):
        """Test --help shows help and exits."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_quiet_flag_parsing(self):
        """Test -q/--quiet flag is parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("-q", "--quiet", action="store_true")

        args = parser.parse_args(["-q"])
        assert args.quiet is True

        args = parser.parse_args(["--quiet"])
        assert args.quiet is True

    def test_monitor_argument_parsing(self):
        """Test --monitor argument parsing."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--monitor", type=int, default=0)

        args = parser.parse_args(["--monitor", "2"])
        assert args.monitor == 2

    def test_quality_argument_parsing(self):
        """Test --quality argument parsing."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--quality", type=int, default=85)

        args = parser.parse_args(["--quality", "50"])
        assert args.quality == 50

    def test_output_argument_parsing(self):
        """Test -o/--output argument parsing."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("-o", "--output", type=str)

        args = parser.parse_args(["-o", "/path/to/output.jpg"])
        assert args.output == "/path/to/output.jpg"


class TestCLIListAction:
    """Test --list action."""

    def test_list_action_returns_zero(self):
        """Test --list returns 0 on success."""
        from scitex.capture.cli import main

        mock_info = {
            "Windows": {
                "Details": [
                    {
                        "ProcessName": "test_app",
                        "Title": "Test Window",
                        "Handle": 12345,
                        "ProcessId": 1234,
                    }
                ]
            }
        }

        with patch("sys.argv", ["capture", "--list"]):
            with patch("scitex.capture.get_info", return_value=mock_info):
                result = main()
                assert result == 0

    def test_list_action_empty_windows(self):
        """Test --list with no windows."""
        from scitex.capture.cli import main

        mock_info = {"Windows": {"Details": []}}

        with patch("sys.argv", ["capture", "--list"]):
            with patch("scitex.capture.get_info", return_value=mock_info):
                result = main()
                assert result == 0


class TestCLIInfoAction:
    """Test --info action."""

    def test_info_action_returns_zero(self):
        """Test --info returns 0 on success."""
        from scitex.capture.cli import main

        mock_info = {
            "Monitors": {
                "Count": 2,
                "PrimaryMonitor": 0,
                "Details": [
                    {
                        "DeviceName": "Display1",
                        "IsPrimary": True,
                        "Bounds": {"Width": 1920, "Height": 1080},
                    }
                ],
            },
            "Windows": {"VisibleCount": 5, "Details": []},
            "VirtualDesktops": {"Supported": True, "Note": "Test"},
        }

        with patch("sys.argv", ["capture", "--info"]):
            with patch("scitex.capture.get_info", return_value=mock_info):
                result = main()
                assert result == 0


class TestCLIStopAction:
    """Test --stop action."""

    def test_stop_action_returns_zero(self):
        """Test --stop returns 0."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "--stop"]):
            with patch("scitex.capture.stop") as mock_stop:
                result = main()
                mock_stop.assert_called_once()
                assert result == 0


class TestCLIGifAction:
    """Test --gif action."""

    def test_gif_action_success(self):
        """Test --gif returns 0 when GIF created."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "--gif"]):
            with patch("scitex.capture.gif", return_value="/path/to/output.gif"):
                result = main()
                assert result == 0

    def test_gif_action_no_session(self):
        """Test --gif returns 1 when no session found."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "--gif"]):
            with patch("scitex.capture.gif", return_value=None):
                result = main()
                assert result == 1


class TestCLIDefaultCapture:
    """Test default capture behavior."""

    def test_default_capture_success(self):
        """Test default capture returns 0 on success."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture"]):
            with patch("scitex.capture.snap", return_value="/path/to/screenshot.jpg"):
                result = main()
                assert result == 0

    def test_default_capture_failure(self):
        """Test default capture returns 1 on failure."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture"]):
            with patch("scitex.capture.snap", return_value=None):
                result = main()
                assert result == 1

    def test_capture_with_message(self):
        """Test capture with positional message argument."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "test message"]):
            with patch("scitex.capture.snap") as mock_snap:
                mock_snap.return_value = "/path/to/screenshot.jpg"
                result = main()

                mock_snap.assert_called_once()
                call_kwargs = mock_snap.call_args[1]
                assert call_kwargs["message"] == "test message"

    def test_capture_with_output(self):
        """Test capture with --output argument."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "-o", "/custom/path.jpg"]):
            with patch("scitex.capture.snap") as mock_snap:
                mock_snap.return_value = "/custom/path.jpg"
                result = main()

                call_kwargs = mock_snap.call_args[1]
                assert call_kwargs["path"] == "/custom/path.jpg"

    def test_capture_with_quality(self):
        """Test capture with --quality argument."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "--quality", "50"]):
            with patch("scitex.capture.snap") as mock_snap:
                mock_snap.return_value = "/path.jpg"
                result = main()

                call_kwargs = mock_snap.call_args[1]
                assert call_kwargs["quality"] == 50

    def test_capture_with_monitor(self):
        """Test capture with --monitor argument."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "--monitor", "1"]):
            with patch("scitex.capture.snap") as mock_snap:
                mock_snap.return_value = "/path.jpg"
                result = main()

                call_kwargs = mock_snap.call_args[1]
                assert call_kwargs["monitor_id"] == 1

    def test_capture_with_all(self):
        """Test capture with --all argument."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "--all"]):
            with patch("scitex.capture.snap") as mock_snap:
                mock_snap.return_value = "/path.jpg"
                result = main()

                call_kwargs = mock_snap.call_args[1]
                assert call_kwargs["all"] is True

    def test_capture_with_app(self):
        """Test capture with --app argument."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "--app", "chrome"]):
            with patch("scitex.capture.snap") as mock_snap:
                mock_snap.return_value = "/path.jpg"
                result = main()

                call_kwargs = mock_snap.call_args[1]
                assert call_kwargs["app"] == "chrome"

    def test_capture_with_url(self):
        """Test capture with --url argument."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "--url", "localhost:8000"]):
            with patch("scitex.capture.snap") as mock_snap:
                mock_snap.return_value = "/path.jpg"
                result = main()

                call_kwargs = mock_snap.call_args[1]
                assert call_kwargs["url"] == "localhost:8000"

    def test_capture_quiet_mode(self):
        """Test capture with --quiet argument."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture", "-q"]):
            with patch("scitex.capture.snap") as mock_snap:
                mock_snap.return_value = "/path.jpg"
                result = main()

                call_kwargs = mock_snap.call_args[1]
                assert call_kwargs["verbose"] is False


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_exception_returns_one(self):
        """Test exception during execution returns 1."""
        from scitex.capture.cli import main

        with patch("sys.argv", ["capture"]):
            with patch("scitex.capture.snap", side_effect=RuntimeError("Test error")):
                result = main()
                assert result == 1


class TestCLIModuleExports:
    """Test module exports."""

    def test_main_importable(self):
        """Test main function can be imported."""
        from scitex.capture.cli import main

        assert main is not None
        assert callable(main)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
