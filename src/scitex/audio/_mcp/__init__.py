#!/usr/bin/env python3
# File: __init__.py
"""MCP server components."""

from .handlers import (
    announce_context_handler,
    check_audio_status_handler,
    clear_audio_cache_handler,
    generate_audio_handler,
    list_audio_files_handler,
    list_backends_handler,
    list_voices_handler,
    play_audio_handler,
    speak_handler,
    speech_queue_status_handler,
)
from .speak_handlers import (
    speak_local_handler,
    speak_relay_handler,
)

__all__ = [
    "speak_handler",
    "speak_local_handler",
    "speak_relay_handler",
    "generate_audio_handler",
    "list_backends_handler",
    "list_voices_handler",
    "play_audio_handler",
    "list_audio_files_handler",
    "clear_audio_cache_handler",
    "check_audio_status_handler",
    "speech_queue_status_handler",
    "announce_context_handler",
]

