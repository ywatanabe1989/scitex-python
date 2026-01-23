#!/usr/bin/env python3
"""Speak handlers for scitex.audio MCP server.

Provides speak_local_handler and speak_relay_handler for explicit control
over audio playback location (server vs relay).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

__all__ = [
    "speak_local_handler",
    "speak_relay_handler",
]


def _get_audio_dir() -> Path:
    """Get the audio output directory."""
    import os

    base_dir = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
    audio_dir = base_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


def _get_signature() -> str:
    """Get signature string with hostname, project, and branch."""
    import os
    import socket
    import subprocess

    hostname = socket.gethostname()
    cwd = os.getcwd()
    project = os.path.basename(cwd)

    branch = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
    except Exception:
        pass

    parts = [f"Hostname: {hostname}", f"Project: {project}"]
    if branch:
        parts.append(f"Branch: {branch}")

    return ". ".join(parts) + ". "


async def speak_local_handler(
    text: str,
    backend: str | None = None,
    voice: str | None = None,
    rate: int = 150,
    speed: float = 1.5,
    play: bool = True,
    save: bool = False,
    fallback: bool = True,
    agent_id: str | None = None,
    signature: bool = False,
) -> dict:
    """Play audio on the LOCAL/SERVER machine.

    Use when running Claude Code directly on your local machine.
    Audio plays on the machine where the MCP server is running.
    """
    try:
        from .. import speak as tts_speak
        from .._cross_process_lock import AudioPlaybackLock

        loop = asyncio.get_event_loop()

        final_text = text
        sig = None
        if signature:
            sig = _get_signature()
            final_text = sig + text

        output_path = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(_get_audio_dir() / f"tts_{timestamp}.mp3")

        def do_speak():
            # Acquire cross-process lock for FIFO audio playback
            lock = AudioPlaybackLock()
            lock.acquire(timeout=120.0)
            try:
                return tts_speak(
                    text=final_text,
                    backend=backend,
                    voice=voice,
                    rate=rate,
                    speed=speed,
                    play=play,
                    output_path=output_path,
                    fallback=fallback,
                    mode="local",  # Force local mode
                )
            finally:
                lock.release()

        result_path = await loop.run_in_executor(None, do_speak)

        result = {
            "success": True,
            "text": text,
            "backend": backend,
            "played": play,
            "played_on": "server",
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
        }
        if signature:
            result["signature"] = sig
            result["full_text"] = final_text
        if result_path:
            result["path"] = str(result_path)

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


async def speak_relay_handler(
    text: str,
    backend: str | None = None,
    voice: str | None = None,
    rate: int = 150,
    speed: float = 1.5,
    play: bool = True,
    save: bool = False,
    fallback: bool = True,
    agent_id: str | None = None,
) -> dict:
    """Forward speech to RELAY server for remote playback.

    Use when running on a remote server and want audio on your local machine.
    Returns detailed error with setup instructions if relay unavailable.
    """
    from .._branding import DEFAULT_PORT, get_relay_url, get_ssh_client_ip
    from .._relay import RelayClient, is_relay_available

    # Get relay URL (auto-detects from SSH_CLIENT if not configured)
    relay_url = get_relay_url()
    ssh_client_ip = get_ssh_client_ip()

    if not relay_url:
        return {
            "success": False,
            "error": "Relay server URL not configured",
            "reason": "No SSH session detected and no env vars set",
            "instructions": [
                "1. Start relay server on your LOCAL machine:",
                f"   scitex audio serve -t http --port {DEFAULT_PORT}",
                "",
                "2. SSH to this server (relay URL auto-detected from SSH_CLIENT)",
                "",
                "3. Or set env var manually:",
                f"   export SCITEX_AUDIO_RELAY_URL=http://YOUR_LOCAL_IP:{DEFAULT_PORT}",
            ],
        }

    # Check if relay server is reachable
    if not is_relay_available():
        source = "auto-detected from SSH_CLIENT" if ssh_client_ip else "from env var"
        return {
            "success": False,
            "error": "Relay server not reachable",
            "reason": f"Cannot connect to {relay_url} ({source})",
            "relay_url": relay_url,
            "auto_detected": ssh_client_ip is not None,
            "ssh_client_ip": ssh_client_ip,
            "instructions": [
                "1. Start relay server on your LOCAL machine:",
                f"   scitex audio serve -t http --port {DEFAULT_PORT}",
                "",
                f"2. Current relay URL: {relay_url}",
                f"   Source: {source}",
                "",
                "3. Test connectivity:",
                f"   curl {relay_url}/health",
            ],
        }

    # Forward to relay server
    try:
        loop = asyncio.get_event_loop()

        def do_relay():
            client = RelayClient(relay_url)
            return client.speak(
                text=text,
                backend=backend,
                voice=voice,
                rate=rate,
                speed=speed,
                play=play,
                save=save,
                fallback=fallback,
                agent_id=agent_id,
            )

        result = await loop.run_in_executor(None, do_relay)

        result["played_on"] = "relay"
        result["relay_url"] = relay_url
        result["timestamp"] = datetime.now().isoformat()

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Relay request failed: {str(e)}",
            "relay_url": relay_url,
            "instructions": [
                "1. Check relay server is still running",
                "2. Check network connectivity",
                f"3. Test: curl -X POST {relay_url}/speak "
                "-H 'Content-Type: application/json' -d '{\"text\": \"test\"}'",
            ],
        }


# EOF
