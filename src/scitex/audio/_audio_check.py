#!/usr/bin/env python3
"""Audio availability checking utilities.

Provides functions to check if local audio playback is available
before attempting to play audio.
"""

from __future__ import annotations

import subprocess

__all__ = ["check_local_audio_available"]


def check_local_audio_available() -> dict:
    """Check if local audio playback is available.

    Checks PulseAudio sink state to determine if audio can actually be heard.
    On NAS or headless servers, the sink is typically SUSPENDED.

    Returns:
        dict with keys:
        - available: bool - True if local audio output is likely to work
        - state: str - 'RUNNING', 'IDLE', 'SUSPENDED', 'NO_SINK', etc.
        - reason: str - Human-readable explanation
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "sinks", "short"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {
                "available": False,
                "state": "NO_PACTL",
                "reason": "PulseAudio not available",
            }

        if not result.stdout.strip():
            return {
                "available": False,
                "state": "NO_SINK",
                "reason": "No audio sinks found",
            }

        # Parse sink state (format: id\tname\tmodule\tformat\tstate)
        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 5:
                state = parts[4]
                if state == "SUSPENDED":
                    return {
                        "available": False,
                        "state": "SUSPENDED",
                        "reason": "Audio sink SUSPENDED (no active output device)",
                    }
                elif state in ("RUNNING", "IDLE"):
                    return {
                        "available": True,
                        "state": state,
                        "reason": f"Audio sink is {state}",
                    }

        return {
            "available": False,
            "state": "UNKNOWN",
            "reason": "Could not determine sink state",
        }

    except FileNotFoundError:
        # No pactl - might be macOS or minimal system, assume available
        return {
            "available": True,
            "state": "NO_PACTL",
            "reason": "pactl not found, assuming audio available",
        }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "state": "TIMEOUT",
            "reason": "PulseAudio query timed out",
        }
    except Exception as e:
        return {
            "available": False,
            "state": "ERROR",
            "reason": str(e),
        }


# EOF
