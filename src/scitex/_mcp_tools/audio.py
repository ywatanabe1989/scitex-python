#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/audio.py
"""Audio module tools for FastMCP unified server."""

from __future__ import annotations

import json
from typing import Optional


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def register_audio_tools(mcp) -> None:
    """Register audio tools with FastMCP server."""

    @mcp.tool()
    async def audio_speak(
        text: str,
        backend: Optional[str] = None,
        voice: Optional[str] = None,
        rate: int = 150,
        speed: float = 1.5,
        play: bool = True,
        save: bool = False,
        fallback: bool = True,
        agent_id: Optional[str] = None,
        wait: bool = True,
        signature: bool = False,
    ) -> str:
        """[audio] Convert text to speech with fallback (pyttsx3 -> gtts -> elevenlabs)."""
        from scitex.audio._mcp.handlers import speak_handler

        result = await speak_handler(
            text=text,
            backend=backend,
            voice=voice,
            rate=rate,
            speed=speed,
            play=play,
            save=save,
            fallback=fallback,
            agent_id=agent_id,
            wait=wait,
            signature=signature,
        )
        return _json(result)

    @mcp.tool()
    async def audio_generate_audio(
        text: str,
        backend: Optional[str] = "gtts",
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
        return_base64: bool = False,
    ) -> str:
        """[audio] Generate speech audio file without playing."""
        from scitex.audio._mcp.handlers import generate_audio_handler

        result = await generate_audio_handler(
            text=text,
            backend=backend,
            voice=voice,
            output_path=output_path,
            return_base64=return_base64,
        )
        return _json(result)

    @mcp.tool()
    async def audio_list_backends() -> str:
        """[audio] List available TTS backends and their status."""
        from scitex.audio._mcp.handlers import list_backends_handler

        result = await list_backends_handler()
        return _json(result)

    @mcp.tool()
    async def audio_list_voices(backend: str = "gtts") -> str:
        """[audio] List available voices for a backend."""
        from scitex.audio._mcp.handlers import list_voices_handler

        result = await list_voices_handler(backend=backend)
        return _json(result)

    @mcp.tool()
    async def audio_play_audio(path: str) -> str:
        """[audio] Play an audio file."""
        from scitex.audio._mcp.handlers import play_audio_handler

        result = await play_audio_handler(path=path)
        return _json(result)

    @mcp.tool()
    async def audio_list_audio_files(limit: int = 20) -> str:
        """[audio] List generated audio files."""
        from scitex.audio._mcp.handlers import list_audio_files_handler

        result = await list_audio_files_handler(limit=limit)
        return _json(result)

    @mcp.tool()
    async def audio_clear_audio_cache(max_age_hours: float = 24) -> str:
        """[audio] Clear generated audio files."""
        from scitex.audio._mcp.handlers import clear_audio_cache_handler

        result = await clear_audio_cache_handler(max_age_hours=max_age_hours)
        return _json(result)

    @mcp.tool()
    async def audio_check_audio_status() -> str:
        """[audio] Check WSL audio connectivity and available playback methods."""
        from scitex.audio._mcp.handlers import check_audio_status_handler

        result = await check_audio_status_handler()
        return _json(result)

    @mcp.tool()
    async def audio_speech_queue_status() -> str:
        """[audio] Get the current speech queue status."""
        from scitex.audio._mcp.handlers import speech_queue_status_handler

        result = await speech_queue_status_handler()
        return _json(result)

    @mcp.tool()
    async def audio_announce_context(include_full_path: bool = False) -> str:
        """[audio] Announce current working directory and git branch."""
        from scitex.audio._mcp.handlers import announce_context_handler

        result = await announce_context_handler(include_full_path=include_full_path)
        return _json(result)

    @mcp.tool()
    async def audio_speak_local(
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
        """[audio] Convert text to speech on the LOCAL/SERVER machine.

        Use when running Claude Code directly on your local machine.
        Audio plays where MCP server runs.
        """
        from scitex.audio._mcp.speak_handlers import speak_local_handler

        result = await speak_local_handler(
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
        return _json(result)

    @mcp.tool()
    async def audio_speak_relay(
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
        """[audio] Convert text to speech via RELAY server (remote playback).

        Use when running on remote server (NAS) and want audio on your
        local machine. Returns error with setup instructions if unavailable.
        """
        from scitex.audio._mcp.speak_handlers import speak_relay_handler

        result = await speak_relay_handler(
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
        return _json(result)


# EOF
