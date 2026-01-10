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
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/capture/cli.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-18 09:55:58 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/capture/cli.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/capture/cli.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# CLI for scitex.capture - AI's Camera
# """
# 
# import argparse
# import sys
# 
# 
# def main():
#     """Main CLI entry point."""
#     parser = argparse.ArgumentParser(
#         description="scitex.capture - AI's Camera: Capture screenshots from anywhere",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python -m scitex.capture                        # Capture current screen
#   python -m scitex.capture --all                  # Capture all monitors
#   python -m scitex.capture --app chrome           # Capture Chrome window
#   python -m scitex.capture --url 127.0.0.1:8000   # Capture URL
#   python -m scitex.capture --monitor 1            # Capture monitor 1
#   python -m scitex.capture --list                 # List available windows
# 
#   python -m scitex.capture --start                # Start monitoring
#   python -m scitex.capture --stop                 # Stop monitoring
#   python -m scitex.capture --gif                  # Create GIF from session
#   python -m scitex.capture --mcp                  # Start MCP server
#         """,
#     )
# 
#     # Capture options
#     parser.add_argument("message", nargs="?", help="Optional message for filename")
#     parser.add_argument("--all", action="store_true", help="Capture all monitors")
#     parser.add_argument("--app", type=str, help="App name to capture (e.g., chrome)")
#     parser.add_argument("--url", type=str, help="URL to capture (e.g., 127.0.0.1:8000)")
#     parser.add_argument("--monitor", type=int, default=0, help="Monitor ID (0-based)")
#     parser.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100)")
#     parser.add_argument("-o", "--output", type=str, help="Output path")
# 
#     # Actions
#     parser.add_argument("--list", action="store_true", help="List available windows")
#     parser.add_argument("--info", action="store_true", help="Show display info")
#     parser.add_argument("--start", action="store_true", help="Start monitoring")
#     parser.add_argument("--stop", action="store_true", help="Stop monitoring")
#     parser.add_argument(
#         "--gif", action="store_true", help="Create GIF from latest session"
#     )
#     parser.add_argument("--mcp", action="store_true", help="Start MCP server")
# 
#     # Options
#     parser.add_argument(
#         "--interval",
#         type=float,
#         default=1.0,
#         help="Monitoring interval in seconds",
#     )
#     parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
# 
#     args = parser.parse_args()
# 
#     # Import scitex.capture after parsing to avoid import overhead for --help
#     from scitex import capture
# 
#     verbose = not args.quiet
# 
#     try:
#         # Handle actions
#         if args.list:
#             info = capture.get_info()
#             windows = info.get("Windows", {}).get("Details", [])
#             print(f"\n📱 Visible Windows ({len(windows)}):")
#             print("=" * 60)
#             for i, win in enumerate(windows, 1):
#                 print(f"{i}. [{win['ProcessName']}] {win['Title']}")
#                 print(f"   Handle: {win['Handle']} | PID: {win['ProcessId']}")
#             return 0
# 
#         elif args.info:
#             info = capture.get_info()
#             monitors = info.get("Monitors", {})
#             windows = info.get("Windows", {})
#             vd = info.get("VirtualDesktops", {})
# 
#             print("\n🖥️  Display Information")
#             print("=" * 60)
#             print(f"\n📺 Monitors: {monitors.get('Count')}")
#             print(f"   Primary: {monitors.get('PrimaryMonitor')}")
# 
#             for i, mon in enumerate(monitors.get("Details", [])):
#                 bounds = mon.get("Bounds", {})
#                 print(f"\n   Monitor {i}:")
#                 print(f"     Device: {mon.get('DeviceName')}")
#                 print(f"     Resolution: {bounds.get('Width')}x{bounds.get('Height')}")
#                 print(f"     Primary: {mon.get('IsPrimary')}")
# 
#             print(f"\n🪟 Windows: {windows.get('VisibleCount')}")
#             print(f"   On current virtual desktop: {len(windows.get('Details', []))}")
# 
#             print(f"\n🖥️  Virtual Desktops:")
#             print(f"   Supported: {vd.get('Supported')}")
#             print(f"   Note: {vd.get('Note')}")
# 
#             return 0
# 
#         elif args.start:
#             print(f"📸 Starting monitoring (interval: {args.interval}s)...")
#             capture.start(
#                 interval=args.interval,
#                 verbose=verbose,
#                 monitor_id=args.monitor,
#                 all=args.all,
#             )
#             print(
#                 "✅ Monitoring started. Press Ctrl+C to stop, or run: python -m scitex.capture --stop"
#             )
#             print(f"📁 Saving to: ~/.scitex/capture/")
# 
#             # Keep running
#             try:
#                 import time
# 
#                 while True:
#                     time.sleep(1)
#             except KeyboardInterrupt:
#                 capture.stop()
#                 print("\n✅ Monitoring stopped")
# 
#             return 0
# 
#         elif args.stop:
#             capture.stop()
#             print("✅ Monitoring stopped")
#             return 0
# 
#         elif args.gif:
#             print("📹 Creating GIF from latest session...")
#             path = capture.gif()
#             if path:
#                 print(f"✅ GIF created: {path}")
#                 return 0
#             else:
#                 print("❌ No session found")
#                 return 1
# 
#         elif args.mcp:
#             print("🤖 Starting scitex.capture MCP server...")
#             print("Add to Claude Code settings:")
#             print("{")
#             print('  "mcpServers": {')
#             print('    "scitex-capture": {')
#             print('      "command": "python",')
#             print('      "args": ["-m", "scitex.capture", "--mcp"]')
#             print("    }")
#             print("  }")
#             print("}")
#             print()
# 
#             # Start MCP server
#             import asyncio
#             from .mcp_server import main as mcp_main
# 
#             asyncio.run(mcp_main())
#             return 0
# 
#         # Default: capture screenshot
#         else:
#             path = capture.snap(
#                 message=args.message,
#                 path=args.output,
#                 quality=args.quality,
#                 monitor_id=args.monitor,
#                 all=args.all,
#                 app=args.app,
#                 url=args.url,
#                 verbose=verbose,
#             )
# 
#             if path:
#                 if not args.quiet:
#                     print(f"✅ {path}")
#                 return 0
#             else:
#                 print("❌ Screenshot failed")
#                 return 1
# 
#     except KeyboardInterrupt:
#         print("\n⚠️  Interrupted")
#         return 130
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return 1
# 
# 
# if __name__ == "__main__":
#     sys.exit(main())
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/capture/cli.py
# --------------------------------------------------------------------------------
