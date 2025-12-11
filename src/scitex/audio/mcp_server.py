#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/mcp_server.py
# ----------------------------------------

"""
MCP Server for SciTeX Audio - Text-to-Speech with Multiple Backends

Fallback order: pyttsx3 -> gtts -> elevenlabs

Backends:
    - pyttsx3: System TTS (offline, free)
    - gtts: Google TTS (free, requires internet)
    - elevenlabs: ElevenLabs (paid, high quality)
"""

from __future__ import annotations

import asyncio
import base64
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

__all__ = ["AudioServer", "main"]


@dataclass
class SpeechRequest:
    """A queued speech request."""

    request_id: str
    text: str
    backend: Optional[str] = None
    voice: Optional[str] = None
    rate: Optional[int] = None
    speed: Optional[float] = None
    play: bool = True
    save: bool = False
    fallback: bool = True
    future: asyncio.Future = field(default_factory=lambda: None)
    created_at: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None  # Track which agent made the request

# Directory configuration
SCITEX_BASE_DIR = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
SCITEX_AUDIO_DIR = SCITEX_BASE_DIR / "audio"


def get_audio_dir() -> Path:
    """Get the audio output directory."""
    audio_dir = SCITEX_AUDIO_DIR
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


class AudioServer:
    """MCP Server for Text-to-Speech with multiple backends.

    Features a sequential speech queue to prevent audio overlap when
    multiple agents request speech simultaneously.
    """

    def __init__(self):
        self.server = Server("scitex-audio")
        # Speech queue for sequential processing
        self._speech_queue: asyncio.Queue[SpeechRequest] = asyncio.Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._current_request: Optional[SpeechRequest] = None
        self._processed_count: int = 0
        self._is_processing: bool = False
        self.setup_handlers()

    async def start_queue_processor(self):
        """Start the background queue processor if not already running."""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(
                self._process_speech_queue()
            )

    async def _process_speech_queue(self):
        """Process speech requests sequentially from the queue."""
        while True:
            try:
                # Wait for next request
                request = await self._speech_queue.get()
                self._current_request = request
                self._is_processing = True

                try:
                    # Execute the speech request
                    result = await self._execute_speak(request)
                    # Set the result on the future if it exists
                    if request.future and not request.future.done():
                        request.future.set_result(result)
                except Exception as e:
                    # Set exception on future if it exists
                    if request.future and not request.future.done():
                        request.future.set_exception(e)
                finally:
                    self._processed_count += 1
                    self._current_request = None
                    self._is_processing = False
                    self._speech_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception:
                # Continue processing even if one request fails
                continue

    async def _execute_speak(self, request: SpeechRequest) -> dict:
        """Execute a single speech request."""
        from . import available_backends
        from . import speak as tts_speak

        loop = asyncio.get_event_loop()

        # Determine output path
        output_path = None
        if request.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(get_audio_dir() / f"tts_{timestamp}.mp3")

        def do_speak():
            return tts_speak(
                text=request.text,
                backend=request.backend,
                voice=request.voice,
                play=request.play,
                output_path=output_path,
                fallback=request.fallback,
                rate=request.rate,
                speed=request.speed,
            )

        result_path = await loop.run_in_executor(None, do_speak)

        backends = available_backends()
        used_backend = request.backend or (backends[0] if backends else None)

        response = {
            "success": True,
            "request_id": request.request_id,
            "text": request.text,
            "backend": used_backend,
            "available_backends": backends,
            "voice": request.voice,
            "played": request.play,
            "fallback_enabled": request.fallback,
            "timestamp": datetime.now().isoformat(),
            "agent_id": request.agent_id,
        }

        if output_path:
            response["saved_to"] = output_path

        return response

    def setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools():
            return [
                types.Tool(
                    name="speak",
                    description="Convert text to speech with fallback (pyttsx3 -> gtts -> elevenlabs). Requests are queued for sequential playback to prevent audio overlap.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to convert to speech",
                            },
                            "backend": {
                                "type": "string",
                                "description": "TTS backend (auto-selects with fallback if not specified)",
                                "enum": ["pyttsx3", "gtts", "elevenlabs"],
                            },
                            "voice": {
                                "type": "string",
                                "description": "Voice/language (gtts: 'en','fr'; elevenlabs: 'rachel','adam')",
                            },
                            "rate": {
                                "type": "integer",
                                "description": "Speech rate in words per minute (pyttsx3 only, default 150, faster=200+)",
                                "default": 150,
                            },
                            "speed": {
                                "type": "number",
                                "description": "Speed multiplier for gtts (1.0=normal, 1.5=faster, 0.7=slower)",
                                "default": 1.5,
                            },
                            "play": {
                                "type": "boolean",
                                "description": "Play audio after generation",
                                "default": True,
                            },
                            "save": {
                                "type": "boolean",
                                "description": "Save audio to file",
                                "default": False,
                            },
                            "fallback": {
                                "type": "boolean",
                                "description": "Try next backend on failure",
                                "default": True,
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "Optional identifier for the agent making the request",
                            },
                            "wait": {
                                "type": "boolean",
                                "description": "Wait for speech to complete before returning (default: True)",
                                "default": True,
                            },
                        },
                        "required": ["text"],
                    },
                ),
                types.Tool(
                    name="generate_audio",
                    description="Generate speech audio file without playing",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to convert to speech",
                            },
                            "backend": {
                                "type": "string",
                                "description": "TTS backend",
                                "enum": ["gtts", "elevenlabs", "pyttsx3"],
                                "default": "gtts",
                            },
                            "voice": {
                                "type": "string",
                                "description": "Voice/language",
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Output file path",
                            },
                            "return_base64": {
                                "type": "boolean",
                                "description": "Return audio as base64",
                                "default": False,
                            },
                        },
                        "required": ["text"],
                    },
                ),
                types.Tool(
                    name="list_backends",
                    description="List available TTS backends and their status",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="list_voices",
                    description="List available voices for a backend",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "backend": {
                                "type": "string",
                                "description": "TTS backend",
                                "enum": ["gtts", "elevenlabs", "pyttsx3"],
                                "default": "gtts",
                            },
                        },
                    },
                ),
                types.Tool(
                    name="play_audio",
                    description="Play an audio file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to audio file",
                            },
                        },
                        "required": ["path"],
                    },
                ),
                types.Tool(
                    name="list_audio_files",
                    description="List generated audio files",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum files to list",
                                "default": 20,
                            },
                        },
                    },
                ),
                types.Tool(
                    name="clear_audio_cache",
                    description="Clear generated audio files",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "max_age_hours": {
                                "type": "number",
                                "description": "Delete files older than N hours (0 = all)",
                                "default": 24,
                            },
                        },
                    },
                ),
                types.Tool(
                    name="speech_queue_status",
                    description="Get the current speech queue status (pending requests, currently playing, etc.)",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="check_audio_status",
                    description="Check WSL audio connectivity and available playback methods",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            # Ensure queue processor is running for speak requests
            if name == "speak":
                await self.start_queue_processor()
                return await self.speak(**arguments)
            elif name == "generate_audio":
                return await self.generate_audio(**arguments)
            elif name == "list_backends":
                return await self.list_backends()
            elif name == "list_voices":
                return await self.list_voices(**arguments)
            elif name == "play_audio":
                return await self.play_audio(**arguments)
            elif name == "list_audio_files":
                return await self.list_audio_files(**arguments)
            elif name == "clear_audio_cache":
                return await self.clear_audio_cache(**arguments)
            elif name == "speech_queue_status":
                return await self.speech_queue_status()
            elif name == "check_audio_status":
                return await self.check_audio_status()
            else:
                raise ValueError(f"Unknown tool: {name}")

        # Provide audio files as resources
        @self.server.list_resources()
        async def handle_list_resources():
            audio_dir = get_audio_dir()
            if not audio_dir.exists():
                return []

            resources = []
            audio_files = sorted(
                list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:20]

            for audio_file in audio_files:
                mtime = datetime.fromtimestamp(audio_file.stat().st_mtime)
                mime_type = (
                    "audio/mpeg" if audio_file.suffix == ".mp3" else "audio/wav"
                )
                resources.append(
                    types.Resource(
                        uri=f"audio://{audio_file.name}",
                        name=audio_file.name,
                        description=f"Audio from {mtime.strftime('%Y-%m-%d %H:%M:%S')}",
                        mimeType=mime_type,
                    )
                )
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str):
            if uri.startswith("audio://"):
                filename = uri.replace("audio://", "")
                filepath = get_audio_dir() / filename

                if filepath.exists():
                    with open(filepath, "rb") as f:
                        content = base64.b64encode(f.read()).decode()

                    mime_type = (
                        "audio/mpeg" if filepath.suffix == ".mp3" else "audio/wav"
                    )
                    return types.ResourceContent(
                        uri=uri, mimeType=mime_type, content=content
                    )
                else:
                    raise ValueError(f"Audio file not found: {filename}")

    async def speak(
        self,
        text: str,
        backend: Optional[str] = None,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        speed: Optional[float] = None,
        play: bool = True,
        save: bool = False,
        fallback: bool = True,
        agent_id: Optional[str] = None,
        wait: bool = True,
    ):
        """Convert text to speech with fallback support.

        Requests are queued for sequential playback to prevent audio overlap
        when multiple agents request speech simultaneously.

        Args:
            text: Text to convert to speech
            backend: TTS backend to use
            voice: Voice/language selection
            rate: Speech rate (pyttsx3 only)
            speed: Speed multiplier (gtts only)
            play: Whether to play audio after generation
            save: Whether to save audio to file
            fallback: Whether to try next backend on failure
            agent_id: Optional identifier for the requesting agent
            wait: Whether to wait for speech to complete (default: True)
        """
        try:
            # Create a unique request ID
            request_id = str(uuid.uuid4())[:8]

            # Create a future to wait for the result
            loop = asyncio.get_event_loop()
            future = loop.create_future()

            # Create the speech request
            request = SpeechRequest(
                request_id=request_id,
                text=text,
                backend=backend,
                voice=voice,
                rate=rate,
                speed=speed,
                play=play,
                save=save,
                fallback=fallback,
                future=future,
                agent_id=agent_id,
            )

            # Add to queue
            await self._speech_queue.put(request)

            queue_position = self._speech_queue.qsize()

            if wait:
                # Wait for the speech to complete
                result = await future
                result["queue_position"] = 0  # Already processed
                return result
            else:
                # Return immediately with queue info
                return {
                    "success": True,
                    "queued": True,
                    "request_id": request_id,
                    "queue_position": queue_position,
                    "text": text,
                    "agent_id": agent_id,
                    "message": f"Request queued at position {queue_position}",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def speech_queue_status(self):
        """Get the current speech queue status."""
        try:
            current = None
            if self._current_request:
                current = {
                    "request_id": self._current_request.request_id,
                    "text": self._current_request.text[:50] + "..."
                    if len(self._current_request.text) > 50
                    else self._current_request.text,
                    "agent_id": self._current_request.agent_id,
                    "created_at": self._current_request.created_at.isoformat(),
                }

            return {
                "success": True,
                "queue_size": self._speech_queue.qsize(),
                "is_processing": self._is_processing,
                "current_request": current,
                "total_processed": self._processed_count,
                "processor_running": self._queue_processor_task is not None
                and not self._queue_processor_task.done(),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def generate_audio(
        self,
        text: str,
        backend: Optional[str] = None,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
        return_base64: bool = False,
    ):
        """Generate audio file without playing."""
        try:
            from . import speak as tts_speak, available_backends

            loop = asyncio.get_event_loop()

            # Determine output path
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(get_audio_dir() / f"tts_{timestamp}.mp3")

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

            # Get file size
            if result_path.exists():
                result["size_kb"] = round(result_path.stat().st_size / 1024, 2)

            if return_base64 and result_path.exists():
                with open(result_path, "rb") as f:
                    result["base64"] = base64.b64encode(f.read()).decode()

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_backends(self):
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

    async def list_voices(self, backend: str = "gtts"):
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

    async def play_audio(self, path: str):
        """Play an audio file."""
        try:
            from ._base import BaseTTS

            path_obj = Path(path)
            if not path_obj.exists():
                return {"success": False, "error": f"File not found: {path}"}

            loop = asyncio.get_event_loop()

            def do_play():
                # Use the base class play method
                BaseTTS._play_audio(None, path_obj)

            await loop.run_in_executor(None, do_play)

            return {
                "success": True,
                "played": str(path_obj),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_audio_files(self, limit: int = 20):
        """List generated audio files."""
        try:
            audio_dir = get_audio_dir()
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
                        "created": datetime.fromtimestamp(
                            f.stat().st_mtime
                        ).isoformat(),
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

    async def clear_audio_cache(self, max_age_hours: float = 24):
        """Clear audio cache."""
        try:
            audio_dir = get_audio_dir()
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

    async def check_audio_status(self):
        """Check WSL audio connectivity and available playback methods."""
        try:
            from . import check_wsl_audio

            status = check_wsl_audio()
            status["success"] = True
            status["timestamp"] = datetime.now().isoformat()
            return status

        except Exception as e:
            return {"success": False, "error": str(e)}


async def main():
    """Main entry point for the MCP server."""
    server = AudioServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scitex-audio",
                server_version="0.2.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())

# EOF
