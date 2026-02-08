#!/usr/bin/env python3
"""Utility functions for audio module."""

import os
import signal
import subprocess
import time


def kill_process_on_port(port: int, verbose: bool = True) -> None:
    """Kill process using the specified port.

    Args:
        port: Port number to free up
        verbose: Whether to print status messages
    """
    try:
        # Find process using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                try:
                    pid_int = int(pid)
                    if verbose:
                        print(f"Killing process {pid_int} on port {port}...")
                    os.kill(pid_int, signal.SIGTERM)
                    # Wait a bit for graceful shutdown
                    time.sleep(0.5)
                    # Force kill if still running
                    try:
                        os.kill(pid_int, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Already dead
                except (ValueError, ProcessLookupError) as e:
                    if verbose:
                        print(f"Warning: Could not kill process {pid}: {e}")
    except FileNotFoundError:
        # lsof not available, try fuser
        try:
            result = subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and verbose:
                print(f"Killed process on port {port}")
        except FileNotFoundError:
            if verbose:
                print(
                    "Warning: Neither lsof nor fuser found. Cannot kill process on port."
                )


__all__ = ["kill_process_on_port"]

# EOF
