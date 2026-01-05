#!/usr/bin/env python3
# Timestamp: "2025-12-27 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/_mcp_handlers.py
# ----------------------------------------

"""Utility handlers for the scitex-audio MCP server."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from pathlib import Path

__all__ = [
    "generate_audio_handler",
    "list_backends_handler",
    "list_voices_handler",
    "play_audio_handler",
    "list_audio_files_handler",
    "clear_audio_cache_handler",
    "check_audio_status_handler",
]


def _get_audio_dir() -> Path:
    """Get the audio output directory."""
    import os

    base_dir = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
    audio_dir = base_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


async def generate_audio_handler(
    text: str,
    backend: str | None = None,
    voice: str | None = None,
    output_path: str | None = None,
    return_base64: bool = False,
) -> dict:
    """Generate audio file without playing."""
    try:
        from . import speak as tts_speak

        loop = asyncio.get_event_loop()

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(_get_audio_dir() / f"tts_{timestamp}.mp3")

        def do_generate():
            return tts_speak(
                text=text,
                backend=backend,
                voice=voice,
                play=False,
                output_path=output_path,
                fallback=True,
            )

        result_path = await loop.run_in_executor(None, do_generate)

        result = {
            "success": True,
            "path": str(result_path),
            "text": text,
            "backend": backend,
            "timestamp": datetime.now().isoformat(),
        }

        if result_path.exists():
            result["size_kb"] = round(result_path.stat().st_size / 1024, 2)

        if return_base64 and result_path.exists():
            with open(result_path, "rb") as f:
                result["base64"] = base64.b64encode(f.read()).decode()

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_backends_handler() -> dict:
    """List available TTS backends."""
    try:
        from . import available_backends

        backends = available_backends()

        info = []
        for b in ["gtts", "elevenlabs", "pyttsx3"]:
            available = b in backends
            desc = {
                "gtts": "Google TTS - Free, requires internet",
                "elevenlabs": "ElevenLabs - Paid, high quality",
                "pyttsx3": "System TTS - Offline, uses espeak/SAPI5",
            }
            info.append(
                {
                    "name": b,
                    "available": available,
                    "description": desc.get(b, ""),
                }
            )

        return {
            "success": True,
            "backends": info,
            "available": backends,
            "default": backends[0] if backends else None,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_voices_handler(backend: str = "gtts") -> dict:
    """List available voices for a backend."""
    try:
        from . import get_tts

        loop = asyncio.get_event_loop()

        def do_list():
            tts = get_tts(backend)
            return tts.get_voices()

        voices = await loop.run_in_executor(None, do_list)

        return {
            "success": True,
            "backend": backend,
            "voices": voices,
            "count": len(voices),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def play_audio_handler(path: str) -> dict:
    """Play an audio file."""
    try:
        from .engines.base import BaseTTS

        path_obj = Path(path)
        if not path_obj.exists():
            return {"success": False, "error": f"File not found: {path}"}

        loop = asyncio.get_event_loop()

        def do_play():
            BaseTTS._play_audio(None, path_obj)

        await loop.run_in_executor(None, do_play)

        return {
            "success": True,
            "played": str(path_obj),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_audio_files_handler(limit: int = 20) -> dict:
    """List generated audio files."""
    try:
        audio_dir = _get_audio_dir()
        if not audio_dir.exists():
            return {"success": True, "files": [], "count": 0}

        audio_files = sorted(
            list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]

        files = []
        for f in audio_files:
            files.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "size_kb": round(f.stat().st_size / 1024, 2),
                    "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                }
            )

        total_size = sum(f.stat().st_size for f in audio_dir.glob("*.*"))

        return {
            "success": True,
            "files": files,
            "count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "audio_dir": str(audio_dir),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def clear_audio_cache_handler(max_age_hours: float = 24) -> dict:
    """Clear audio cache."""
    try:
        audio_dir = _get_audio_dir()
        if not audio_dir.exists():
            return {"success": True, "deleted": 0}

        deleted = 0
        now = datetime.now()

        for f in list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav")):
            try:
                if max_age_hours == 0:
                    f.unlink()
                    deleted += 1
                else:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    age_hours = (now - mtime).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        f.unlink()
                        deleted += 1
            except Exception:
                pass

        return {
            "success": True,
            "deleted": deleted,
            "max_age_hours": max_age_hours,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def check_audio_status_handler() -> dict:
    """Check WSL audio connectivity and available playback methods."""
    try:
        from . import check_wsl_audio

        status = check_wsl_audio()
        status["success"] = True
        status["timestamp"] = datetime.now().isoformat()
        return status

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
