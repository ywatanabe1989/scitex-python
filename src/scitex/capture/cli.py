#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-18 09:55:58 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/capture/cli.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/capture/cli.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
CLI for scitex.capture - AI's Camera
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="scitex.capture - AI's Camera: Capture screenshots from anywhere",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scitex.capture                        # Capture current screen
  python -m scitex.capture --all                  # Capture all monitors
  python -m scitex.capture --app chrome           # Capture Chrome window
  python -m scitex.capture --url 127.0.0.1:8000   # Capture URL
  python -m scitex.capture --monitor 1            # Capture monitor 1
  python -m scitex.capture --list                 # List available windows

  python -m scitex.capture --start                # Start monitoring
  python -m scitex.capture --stop                 # Stop monitoring
  python -m scitex.capture --gif                  # Create GIF from session
  python -m scitex.capture --mcp                  # Start MCP server
        """,
    )

    # Capture options
    parser.add_argument("message", nargs="?", help="Optional message for filename")
    parser.add_argument("--all", action="store_true", help="Capture all monitors")
    parser.add_argument("--app", type=str, help="App name to capture (e.g., chrome)")
    parser.add_argument("--url", type=str, help="URL to capture (e.g., 127.0.0.1:8000)")
    parser.add_argument("--monitor", type=int, default=0, help="Monitor ID (0-based)")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100)")
    parser.add_argument("-o", "--output", type=str, help="Output path")

    # Actions
    parser.add_argument("--list", action="store_true", help="List available windows")
    parser.add_argument("--info", action="store_true", help="Show display info")
    parser.add_argument("--start", action="store_true", help="Start monitoring")
    parser.add_argument("--stop", action="store_true", help="Stop monitoring")
    parser.add_argument(
        "--gif", action="store_true", help="Create GIF from latest session"
    )
    parser.add_argument("--mcp", action="store_true", help="Start MCP server")

    # Options
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Monitoring interval in seconds",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    # Import scitex.capture after parsing to avoid import overhead for --help
    from scitex import capture

    verbose = not args.quiet

    try:
        # Handle actions
        if args.list:
            info = capture.get_info()
            windows = info.get("Windows", {}).get("Details", [])
            print(f"\n📱 Visible Windows ({len(windows)}):")
            print("=" * 60)
            for i, win in enumerate(windows, 1):
                print(f"{i}. [{win['ProcessName']}] {win['Title']}")
                print(f"   Handle: {win['Handle']} | PID: {win['ProcessId']}")
            return 0

        elif args.info:
            info = capture.get_info()
            monitors = info.get("Monitors", {})
            windows = info.get("Windows", {})
            vd = info.get("VirtualDesktops", {})

            print("\n🖥️  Display Information")
            print("=" * 60)
            print(f"\n📺 Monitors: {monitors.get('Count')}")
            print(f"   Primary: {monitors.get('PrimaryMonitor')}")

            for i, mon in enumerate(monitors.get("Details", [])):
                bounds = mon.get("Bounds", {})
                print(f"\n   Monitor {i}:")
                print(f"     Device: {mon.get('DeviceName')}")
                print(f"     Resolution: {bounds.get('Width')}x{bounds.get('Height')}")
                print(f"     Primary: {mon.get('IsPrimary')}")

            print(f"\n🪟 Windows: {windows.get('VisibleCount')}")
            print(f"   On current virtual desktop: {len(windows.get('Details', []))}")

            print(f"\n🖥️  Virtual Desktops:")
            print(f"   Supported: {vd.get('Supported')}")
            print(f"   Note: {vd.get('Note')}")

            return 0

        elif args.start:
            print(f"📸 Starting monitoring (interval: {args.interval}s)...")
            capture.start(
                interval=args.interval,
                verbose=verbose,
                monitor_id=args.monitor,
                all=args.all,
            )
            print(
                "✅ Monitoring started. Press Ctrl+C to stop, or run: python -m scitex.capture --stop"
            )
            print(f"📁 Saving to: ~/.scitex/capture/")

            # Keep running
            try:
                import time

                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                capture.stop()
                print("\n✅ Monitoring stopped")

            return 0

        elif args.stop:
            capture.stop()
            print("✅ Monitoring stopped")
            return 0

        elif args.gif:
            print("📹 Creating GIF from latest session...")
            path = capture.gif()
            if path:
                print(f"✅ GIF created: {path}")
                return 0
            else:
                print("❌ No session found")
                return 1

        elif args.mcp:
            print("🤖 Starting scitex.capture MCP server...")
            print("Add to Claude Code settings:")
            print("{")
            print('  "mcpServers": {')
            print('    "scitex-capture": {')
            print('      "command": "python",')
            print('      "args": ["-m", "scitex.capture", "--mcp"]')
            print("    }")
            print("  }")
            print("}")
            print()

            # Start MCP server
            import asyncio
            from .mcp_server import main as mcp_main

            asyncio.run(mcp_main())
            return 0

        # Default: capture screenshot
        else:
            path = capture.snap(
                message=args.message,
                path=args.output,
                quality=args.quality,
                monitor_id=args.monitor,
                all=args.all,
                app=args.app,
                url=args.url,
                verbose=verbose,
            )

            if path:
                if not args.quiet:
                    print(f"✅ {path}")
                return 0
            else:
                print("❌ Screenshot failed")
                return 1

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# EOF
