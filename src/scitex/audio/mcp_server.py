#!/usr/bin/env python3
# Timestamp: "2025-12-27 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/mcp_server.py
# ----------------------------------------

"""
MCP Server for SciTeX Audio - Text-to-Speech with Multiple Backends

Uses cross-process FIFO locking to ensure only one instance
plays audio at a time across all Claude Code sessions.
"""

from __future__ import annotations

import asyncio
import base64
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

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
    backend: str | None = None
    voice: str | None = None
    rate: int | None = None
    speed: float | None = None
    play: bool = True
    save: bool = False
    fallback: bool = True
    future: asyncio.Future = field(default_factory=lambda: None)
    created_at: datetime = field(default_factory=datetime.now)
    agent_id: str | None = None


# Directory configuration
SCITEX_BASE_DIR = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
SCITEX_AUDIO_DIR = SCITEX_BASE_DIR / "audio"


def get_audio_dir() -> Path:
    """Get the audio output directory."""
    SCITEX_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return SCITEX_AUDIO_DIR


class AudioServer:
    """MCP Server for Text-to-Speech with cross-process FIFO queuing."""

    def __init__(self):
        self.server = Server("scitex-audio")
        self._speech_queue: asyncio.Queue[SpeechRequest] = asyncio.Queue()
        self._queue_processor_task: asyncio.Task | None = None
        self._current_request: SpeechRequest | None = None
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
                request = await self._speech_queue.get()
                self._current_request = request
                self._is_processing = True

                try:
                    result = await self._execute_speak(request)
                    if request.future and not request.future.done():
                        request.future.set_result(result)
                except Exception as e:
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
                continue

    async def _execute_speak(self, request: SpeechRequest) -> dict:
        """Execute a single speech request with cross-process locking."""
        from . import available_backends
        from . import speak as tts_speak
        from ._cross_process_lock import AudioPlaybackLock

        loop = asyncio.get_event_loop()

        output_path = None
        if request.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(get_audio_dir() / f"tts_{timestamp}.mp3")

        def do_speak_with_lock():
            # Acquire cross-process lock to ensure FIFO across all instances
            lock = AudioPlaybackLock()
            lock.acquire(timeout=120.0)
            try:
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
            finally:
                lock.release()

        await loop.run_in_executor(None, do_speak_with_lock)

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
        """Set up MCP server handlers."""
        from ._mcp_handlers import (
            check_audio_status_handler,
            clear_audio_cache_handler,
            generate_audio_handler,
            list_audio_files_handler,
            list_backends_handler,
            list_voices_handler,
            play_audio_handler,
        )
        from ._mcp_tool_schemas import get_tool_schemas

        @self.server.list_tools()
        async def handle_list_tools():
            return get_tool_schemas()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            if name == "speak":
                await self.start_queue_processor()
                return await self.speak(**arguments)
            elif name == "generate_audio":
                return await generate_audio_handler(**arguments)
            elif name == "list_backends":
                return await list_backends_handler()
            elif name == "list_voices":
                return await list_voices_handler(**arguments)
            elif name == "play_audio":
                return await play_audio_handler(**arguments)
            elif name == "list_audio_files":
                return await list_audio_files_handler(**arguments)
            elif name == "clear_audio_cache":
                return await clear_audio_cache_handler(**arguments)
            elif name == "speech_queue_status":
                return await self.speech_queue_status()
            elif name == "check_audio_status":
                return await check_audio_status_handler()
            elif name == "announce_context":
                return await self.announce_context(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

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
                mime_type = "audio/mpeg" if audio_file.suffix == ".mp3" else "audio/wav"
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
        backend: str | None = None,
        voice: str | None = None,
        rate: int | None = None,
        speed: float | None = None,
        play: bool = True,
        save: bool = False,
        fallback: bool = True,
        agent_id: str | None = None,
        wait: bool = True,
    ):
        """Queue speech request with cross-process FIFO ordering."""
        try:
            request_id = str(uuid.uuid4())[:8]
            loop = asyncio.get_event_loop()
            future = loop.create_future()

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

            await self._speech_queue.put(request)
            queue_position = self._speech_queue.qsize()

            if wait:
                result = await future
                result["queue_position"] = 0
                return result
            else:
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

    async def announce_context(self, include_full_path: bool = False):
        """Announce the current working directory and git branch."""
        import subprocess

        try:
            # Get current working directory
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

            # Speak the announcement
            await self.start_queue_processor()
            result = await self.speak(text=text)

            return {
                "success": True,
                "directory": str(cwd),
                "directory_name": cwd.name,
                "git_branch": git_branch,
                "announced_text": text,
                "speak_result": result,
            }

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
                server_version="0.3.0",  # Bumped for cross-process FIFO
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())


# EOF
