#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/mcp_server.py
# ----------------------------------------

"""
FastMCP Server for SciTeX Audio - HTTP/SSE Transport Support

Enables remote agents to connect and play audio on local speakers.

Usage:
    scitex audio serve                           # stdio (default)
    scitex audio serve -t http --port 8084       # HTTP transport
    scitex audio serve -t sse --port 8084        # SSE transport

For remote audio:
    1. Run locally:  scitex audio serve -t http --port 8084
    2. SSH tunnel:   ssh -R 8084:localhost:8084 remote-host
    3. Remote agent connects to http://localhost:8084
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Graceful FastMCP dependency handling
try:
    from fastmcp import FastMCP

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None  # type: ignore

__all__ = ["mcp", "run_server", "main", "FASTMCP_AVAILABLE"]

# Initialize MCP server
if FASTMCP_AVAILABLE:
    mcp = FastMCP(
        name="scitex-audio",
        instructions=(
            "Text-to-Speech server with multiple backends (pyttsx3, gtts, elevenlabs). "
            "Use speak() to convert text to speech. Supports cross-process FIFO queuing."
        ),
    )
else:
    mcp = None


def _get_audio_dir() -> Path:
    """Get the audio output directory."""
    import os

    base_dir = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
    audio_dir = base_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


if FASTMCP_AVAILABLE:

    @mcp.tool()
    def speak(
        text: str,
        backend: Optional[str] = None,
        voice: Optional[str] = None,
        rate: int = 150,
        speed: float = 1.5,
        play: bool = True,
        save: bool = False,
        fallback: bool = True,
        agent_id: Optional[str] = None,
    ) -> str:
        """Convert text to speech with fallback (pyttsx3 -> gtts -> elevenlabs).

        Args:
            text: Text to convert to speech
            backend: TTS backend (auto-selects with fallback if not specified)
            voice: Voice/language (gtts: 'en','fr'; elevenlabs: 'rachel','adam')
            rate: Speech rate in words per minute (pyttsx3 only, default 150)
            speed: Speed multiplier for gtts (1.0=normal, 1.5=faster, 0.7=slower)
            play: Play audio after generation (default True)
            save: Save audio to file (default False)
            fallback: Try next backend on failure (default True)
            agent_id: Optional identifier for the agent making the request

        Returns
        -------
            JSON string with success status and details

        Examples
        --------
            speak("Hello world")
            speak("Bonjour", backend="gtts", voice="fr")
            speak("Fast speech", rate=200)
        """
        from . import speak as tts_speak
        from ._cross_process_lock import AudioPlaybackLock

        output_path = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(_get_audio_dir() / f"tts_{timestamp}.mp3")

        try:
            # Acquire cross-process lock to ensure FIFO across all instances
            lock = AudioPlaybackLock()
            lock.acquire(timeout=120.0)
            try:
                tts_speak(
                    text=text,
                    backend=backend,
                    voice=voice,
                    play=play,
                    output_path=output_path,
                    fallback=fallback,
                    rate=rate,
                    speed=speed,
                )
            finally:
                lock.release()

            response = {
                "success": True,
                "text": text,
                "backend": backend,
                "voice": voice,
                "played": play,
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            if output_path:
                response["saved_to"] = output_path

            return json.dumps(response, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    @mcp.tool()
    def list_backends() -> str:
        """List available TTS backends and their status.

        Returns
        -------
            JSON string with available backends and fallback order
        """
        try:
            from . import FALLBACK_ORDER, available_backends

            backends = available_backends()
            return json.dumps(
                {
                    "success": True,
                    "available": backends,
                    "fallback_order": FALLBACK_ORDER,
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    @mcp.tool()
    def check_audio_status() -> str:
        """Check WSL audio connectivity and available playback methods.

        Returns
        -------
            JSON string with audio status information
        """
        try:
            from . import check_wsl_audio

            status = check_wsl_audio()
            status["success"] = True
            status["timestamp"] = datetime.now().isoformat()
            return json.dumps(status, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    @mcp.tool()
    def announce_context(include_full_path: bool = False) -> str:
        """Announce the current working directory and git branch.

        Useful for orientation when starting work in a new session.

        Args:
            include_full_path: Include full path or just directory name

        Returns
        -------
            JSON string with context information and speak result
        """
        import subprocess

        try:
            cwd = Path.cwd()
            dir_name = str(cwd) if include_full_path else cwd.name

            # Try to get git branch
            git_branch = None
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(cwd),
                )
                if result.returncode == 0:
                    git_branch = result.stdout.strip()
            except Exception:
                pass

            # Build announcement text
            if git_branch:
                text = f"Working in {dir_name}, on branch {git_branch}"
            else:
                text = f"Working in {dir_name}"

            # Speak the announcement using the speak tool
            speak_result = speak(text=text)

            return json.dumps(
                {
                    "success": True,
                    "directory": str(cwd),
                    "directory_name": cwd.name,
                    "git_branch": git_branch,
                    "announced_text": text,
                    "speak_result": json.loads(speak_result),
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)


def run_server(
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 8084,
) -> None:
    """Run the MCP server.

    Args:
        transport: Transport protocol ("stdio", "sse", or "http")
        host: Host for HTTP/SSE transport
        port: Port for HTTP/SSE transport
    """
    if not FASTMCP_AVAILABLE:
        import sys

        print("=" * 60)
        print("MCP Server 'scitex-audio' requires the 'fastmcp' package.")
        print()
        print("Install with:")
        print("  pip install fastmcp")
        print()
        print("Or install scitex with MCP support:")
        print("  pip install scitex[mcp]")
        print("=" * 60)
        sys.exit(1)

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        print(f"Starting scitex-audio MCP server (SSE) on {host}:{port}")
        mcp.run(transport="sse", host=host, port=port)
    elif transport == "http":
        print(f"Starting scitex-audio MCP server (HTTP) on {host}:{port}")
        print(f"Connect via: http://{host}:{port}/mcp")
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        raise ValueError(f"Unknown transport: {transport}")


def main():
    """Entry point for scitex-audio-mcp command."""
    run_server(transport="stdio")


if __name__ == "__main__":
    main()


# EOF
