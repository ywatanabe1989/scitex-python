#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/engines/elevenlabs_engine.py
# ----------------------------------------

"""
ElevenLabs TTS backend - High quality, requires API key and payment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .base import BaseTTS

__all__ = ["ElevenLabsTTS"]


class ElevenLabsTTS(BaseTTS):
    """ElevenLabs TTS backend.

    High-quality voices but requires API key and has usage costs.

    Environment:
        ELEVENLABS_API_KEY: Your ElevenLabs API key
    """

    VOICES = {
        "rachel": "21m00Tcm4TlvDq8ikWAM",
        "adam": "pNInz6obpgDQGcFmaJgB",
        "antoni": "ErXwobaYiN019PkySvjV",
        "bella": "EXAVITQu4vr4xnSDxMaL",
        "domi": "AZnzlk1XvdvUeBnXmlld",
        "elli": "MF3mGyEYCl7XYWbV9V6O",
        "josh": "TxGEqnHWrfWFTfGW9XjX",
        "sam": "yoZ06aMxZJJ28mfd3POQ",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "rachel",
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        speed: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.voice = voice
        self.model_id = model_id
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.speed = speed
        self._client = None

    @property
    def name(self) -> str:
        return "elevenlabs"

    @property
    def requires_api_key(self) -> bool:
        return True

    @property
    def requires_internet(self) -> bool:
        return True

    @property
    def client(self):
        """Lazy-load ElevenLabs client."""
        if self._client is None:
            try:
                from elevenlabs.client import ElevenLabs

                self._client = ElevenLabs(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "elevenlabs package not installed. "
                    "Install with: pip install elevenlabs"
                )
        return self._client

    def _get_voice_id(self, voice: Optional[str] = None) -> str:
        """Get voice ID from name or return as-is if already an ID."""
        v = voice or self.voice
        normalized = v.lower()
        return self.VOICES.get(normalized, v)

    def synthesize(self, text: str, output_path: str) -> Path:
        """Synthesize text using ElevenLabs API."""
        voice_id = self._get_voice_id(self.config.get("voice"))

        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=self.model_id,
            voice_settings={
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "speed": self.speed,
            },
            output_format="mp3_44100_128",
        )

        out_path = Path(output_path)
        with open(out_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        return out_path

    def get_voices(self) -> List[dict]:
        """Get available voices."""
        # Start with preset voices
        voices = [
            {"name": name, "id": vid, "type": "preset"}
            for name, vid in self.VOICES.items()
        ]

        # Try to get custom voices
        try:
            response = self.client.voices.get_all()
            for v in response.voices:
                voices.append(
                    {
                        "name": v.name,
                        "id": v.voice_id,
                        "type": "custom",
                        "labels": v.labels,
                    }
                )
        except Exception:
            pass

        return voices


# EOF
