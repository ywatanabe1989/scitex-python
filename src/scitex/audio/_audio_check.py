#!/usr/bin/env python3
"""Audio availability checking utilities.

Provides functions to check if local audio playback is available
before attempting to play audio.
"""

from __future__ import annotations

import os
import shutil
import subprocess

__all__ = ["check_local_audio_available", "check_wsl_windows_audio_available"]


def check_wsl_windows_audio_available() -> dict:
    """Check if WSL Windows audio playback is available.

    In WSL, audio can be played via PowerShell's System.Media.SoundPlayer
    even when PulseAudio is unavailable or SUSPENDED.

    Returns
    -------
        dict with keys:
        - available: bool - True if Windows playback via PowerShell is possible
        - reason: str - Human-readable explanation
    """
    if not os.path.exists("/mnt/c/Windows"):
        return {"available": False, "reason": "Not running in WSL"}

    if not shutil.which("powershell.exe"):
        return {"available": False, "reason": "powershell.exe not found in PATH"}

    return {
        "available": True,
        "reason": "WSL Windows playback available via PowerShell",
    }


def _try_wsl_fallback(
    state: str, reason: str, pulseaudio_state: str | None = None
) -> dict:
    """Try WSL Windows fallback, return appropriate result dict."""
    wsl_check = check_wsl_windows_audio_available()
    if wsl_check["available"]:
        result = {
            "available": True,
            "state": "WSL_WINDOWS",
            "reason": wsl_check["reason"],
            "fallback": "windows_powershell",
        }
        if pulseaudio_state:
            result["pulseaudio_state"] = pulseaudio_state
        return result
    return {"available": False, "state": state, "reason": reason}


def _parse_pulseaudio_state(output: str) -> dict:
    """Parse PulseAudio sink state from pactl output."""
    for line in output.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 5:
            state = parts[4]
            if state == "SUSPENDED":
                return _try_wsl_fallback(
                    "SUSPENDED",
                    "Audio sink SUSPENDED (no active output device)",
                    pulseaudio_state="SUSPENDED",
                )
            if state in ("RUNNING", "IDLE"):
                return {
                    "available": True,
                    "state": state,
                    "reason": f"Audio sink is {state}",
                }

    return _try_wsl_fallback("UNKNOWN", "Could not determine sink state")


def check_local_audio_available() -> dict:
    """Check if local audio playback is available.

    Checks PulseAudio sink state to determine if audio can actually be heard.
    On NAS or headless servers, the sink is typically SUSPENDED.

    In WSL environments, also checks for Windows playback fallback via PowerShell.

    Returns
    -------
        dict with keys:
        - available: bool - True if local audio output is likely to work
        - state: str - 'RUNNING', 'IDLE', 'SUSPENDED', 'NO_SINK', etc.
        - reason: str - Human-readable explanation
        - fallback: str (optional) - Fallback method if primary unavailable
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "sinks", "short"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return _try_wsl_fallback("NO_PACTL", "PulseAudio not available")

        if not result.stdout.strip():
            return _try_wsl_fallback("NO_SINK", "No audio sinks found")

        return _parse_pulseaudio_state(result.stdout)

    except FileNotFoundError:
        wsl_check = check_wsl_windows_audio_available()
        if wsl_check["available"]:
            return {
                "available": True,
                "state": "WSL_WINDOWS",
                "reason": wsl_check["reason"],
                "fallback": "windows_powershell",
            }
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
        return {"available": False, "state": "ERROR", "reason": str(e)}


# EOF
