#!/usr/bin/env python3
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

# Import engines (fail gracefully if dependencies missing)
# Note: BaseTTS and TTSBackend are internal - import from ._base if needed
try:
    from ._pyttsx3_engine import SystemTTS
except ImportError:
    SystemTTS = None

try:
    from ._gtts_engine import GoogleTTS
except ImportError:
    GoogleTTS = None

try:
    from ._elevenlabs_engine import ElevenLabsTTS
except ImportError:
    ElevenLabsTTS = None

__all__ = [
    "SystemTTS",
    "GoogleTTS",
    "ElevenLabsTTS",
]

# EOF
