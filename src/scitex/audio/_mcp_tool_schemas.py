#!/usr/bin/env python3
# Timestamp: "2025-12-27 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/_mcp_tool_schemas.py
# ----------------------------------------

"""Tool schemas for the scitex-audio MCP server."""

from __future__ import annotations

import mcp.types as types

__all__ = ["get_tool_schemas"]


def get_tool_schemas() -> list[types.Tool]:
    """Return all tool schemas for the MCP server."""
    return [
        types.Tool(
            name="speak",
            description=(
                "Convert text to speech with fallback (pyttsx3 -> gtts -> elevenlabs). "
                "Requests are queued for sequential playback to prevent audio overlap."
            ),
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
        types.Tool(
            name="announce_context",
            description=(
                "Announce the current working directory and git branch (if in a git repo). "
                "Useful for orientation when starting work in a new session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "include_full_path": {
                        "type": "boolean",
                        "description": "Include full path or just directory name",
                        "default": False,
                    },
                },
            },
        ),
    ]


# EOF
