#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/engines/pyttsx3_engine.py
# ----------------------------------------

"""
System TTS backend using pyttsx3 - Offline, uses system voices.

Requirements:
    - pip install pyttsx3
    - Linux: sudo apt install espeak-ng libespeak1
    - Windows: Uses SAPI5 (built-in)
    - macOS: Uses NSSpeechSynthesizer (built-in)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .base import BaseTTS

__all__ = ["SystemTTS"]


class SystemTTS(BaseTTS):
    """System TTS backend using pyttsx3.

    Works offline using system's built-in TTS engine.
    Quality varies by platform and available voices.

    Platforms:
        - Linux: espeak/espeak-ng
        - Windows: SAPI5
        - macOS: NSSpeechSynthesizer
    """

    def __init__(
        self,
        rate: int = 150,  # Words per minute
        volume: float = 1.0,  # 0.0 to 1.0
        voice: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rate = rate
        self.volume = volume
        self.voice = voice
        self._engine = None

    @property
    def name(self) -> str:
        return "pyttsx3"

    @property
    def engine(self):
        """Lazy-load pyttsx3 engine."""
        if self._engine is None:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                self._engine.setProperty("volume", self.volume)

                if self.voice:
                    self._set_voice(self.voice)
            except ImportError:
                raise ImportError(
                    "pyttsx3 package not installed. "
                    "Install with: pip install pyttsx3\n"
                    "Linux also requires: sudo apt install espeak-ng libespeak1"
                )
            except RuntimeError as e:
                if "eSpeak" in str(e):
                    raise RuntimeError(
                        "espeak not installed. "
                        "Install with: sudo apt install espeak-ng libespeak1"
                    )
                raise
        return self._engine

    def _set_voice(self, voice_name: str):
        """Set voice by name or ID."""
        voices = self.engine.getProperty("voices")
        for v in voices:
            if voice_name.lower() in v.name.lower() or voice_name == v.id:
                self.engine.setProperty("voice", v.id)
                return
        # If not found, keep default

    def synthesize(self, text: str, output_path: str) -> Path:
        """Synthesize text using system TTS."""
        # Set voice if specified in config
        voice = self.config.get("voice")
        if voice:
            self._set_voice(voice)

        out_path = Path(output_path)

        # pyttsx3 can save to file
        self.engine.save_to_file(text, str(out_path))
        self.engine.runAndWait()

        return out_path

    def speak_direct(self, text: str):
        """Speak directly without saving to file (faster)."""
        voice = self.config.get("voice")
        if voice:
            self._set_voice(voice)

        self.engine.say(text)
        self.engine.runAndWait()

    def get_voices(self) -> List[dict]:
        """Get available system voices."""
        voices = self.engine.getProperty("voices")
        return [
            {
                "name": v.name,
                "id": v.id,
                "type": "system",
                "languages": getattr(v, "languages", []),
            }
            for v in voices
        ]


# EOF
