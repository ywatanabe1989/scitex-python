#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/utils.py
"""Port management utilities for Flask editor."""

import socket
import subprocess
import sys


def find_available_port(start_port: int = 5050, max_attempts: int = 10) -> int:
    """Find an available port, starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue

    raise RuntimeError(
        f"Could not find available port in range {start_port}-{start_port + max_attempts}"
    )


def kill_process_on_port(port: int) -> bool:
    """Try to kill process using the specified port. Returns True if successful."""
    try:
        if sys.platform == "win32":
            # Windows: netstat + taskkill
            result = subprocess.run(
                f"netstat -ano | findstr :{port}",
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        subprocess.run(
                            f"taskkill /F /PID {pid}", shell=True, capture_output=True
                        )
                return True
        else:
            # Linux/Mac: fuser or lsof
            result = subprocess.run(
                ["fuser", "-k", f"{port}/tcp"], capture_output=True, text=True
            )
            if result.returncode == 0:
                return True

            # Fallback to lsof
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"], capture_output=True, text=True
            )
            if result.stdout:
                for pid in result.stdout.strip().split("\n"):
                    if pid:
                        subprocess.run(["kill", "-9", pid], capture_output=True)
                return True
    except Exception:
        pass

    return False


def check_port_available(port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


# EOF
