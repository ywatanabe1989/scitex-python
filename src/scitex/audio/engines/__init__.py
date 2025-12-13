#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/engines/__init__.py
# ----------------------------------------

"""
TTS Engine Backends

Fallback order: pyttsx3 -> gtts -> elevenlabs

Engines:
    - SystemTTS (pyttsx3): Offline, free, uses system TTS
    - GoogleTTS (gtts): Free, requires internet
    - ElevenLabsTTS: Paid, high quality
"""

from .base import BaseTTS, TTSBackend

# Import engines (fail gracefully if dependencies missing)
try:
    from .pyttsx3_engine import SystemTTS
except ImportError:
    SystemTTS = None

try:
    from .gtts_engine import GoogleTTS
except ImportError:
    GoogleTTS = None

try:
    from .elevenlabs_engine import ElevenLabsTTS
except ImportError:
    ElevenLabsTTS = None

__all__ = [
    "BaseTTS",
    "TTSBackend",
    "SystemTTS",
    "GoogleTTS",
    "ElevenLabsTTS",
]

# EOF
