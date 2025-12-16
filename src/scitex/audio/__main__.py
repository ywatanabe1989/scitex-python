#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/__main__.py
# ----------------------------------------

"""
CLI entry point for SciTeX Audio.

Usage:
    python -m scitex.audio --mcp         # Start MCP server
    python -m scitex.audio speak "Hello" # Quick TTS with fallback
    python -m scitex.audio --help        # Show help
"""

import argparse
import asyncio
import sys


def main():
    parser = argparse.ArgumentParser(
        description="SciTeX Audio - Text-to-Speech with fallback (pyttsx3 -> gtts -> elevenlabs)"
    )

    # Global options
    parser.add_argument(
        "--mcp", action="store_true",
        help="Start MCP server (for Claude Code integration)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Speak command
    speak_parser = subparsers.add_parser("speak", help="Quick text-to-speech")
    speak_parser.add_argument("text", help="Text to speak")
    speak_parser.add_argument(
        "-b", "--backend",
        choices=["pyttsx3", "gtts", "elevenlabs"],
        help="TTS backend (auto-selects with fallback if not specified)"
    )
    speak_parser.add_argument(
        "-v", "--voice", help="Voice name or language code"
    )
    speak_parser.add_argument(
        "-o", "--output", help="Save to file"
    )
    speak_parser.add_argument(
        "--no-play", action="store_true", help="Don't play audio"
    )
    speak_parser.add_argument(
        "--no-fallback", action="store_true", help="Disable fallback"
    )

    # List backends
    backends_parser = subparsers.add_parser("backends", help="List available backends")

    # List voices
    voices_parser = subparsers.add_parser("voices", help="List available voices")
    voices_parser.add_argument(
        "-b", "--backend",
        choices=["pyttsx3", "gtts", "elevenlabs"],
        help="Backend to list voices for"
    )

    args = parser.parse_args()

    # MCP server mode
    if args.mcp:
        from .mcp_server import main as server_main
        asyncio.run(server_main())
        return

    if args.command == "speak":
        from . import speak

        speak(
            text=args.text,
            backend=args.backend,
            voice=args.voice,
            play=not args.no_play,
            output_path=args.output,
            fallback=not args.no_fallback,
        )

    elif args.command == "backends":
        from . import available_backends, FALLBACK_ORDER

        backends = available_backends()
        print("Available TTS backends (in fallback order):")
        for b in FALLBACK_ORDER:
            status = "available" if b in backends else "not available"
            desc = {
                "pyttsx3": "System TTS (offline, free)",
                "gtts": "Google TTS (free, needs internet)",
                "elevenlabs": "ElevenLabs (paid, high quality)",
            }
            marker = "[*]" if b in backends else "[ ]"
            print(f"  {marker} {b}: {desc.get(b, '')} - {status}")

    elif args.command == "voices":
        from . import get_tts, available_backends

        backend = args.backend
        if not backend:
            backends = available_backends()
            backend = backends[0] if backends else None

        if backend:
            try:
                tts = get_tts(backend)
                voices = tts.get_voices()
                print(f"Voices for {backend}:")
                for v in voices:
                    print(f"  {v.get('name', 'unknown')}: {v.get('id', '')}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No backends available")

    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()

# EOF
