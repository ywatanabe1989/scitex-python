#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/engines/gtts_engine.py
# ----------------------------------------

"""
Google Text-to-Speech (gTTS) backend - Free, requires internet.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional

from .base import BaseTTS

__all__ = ["GoogleTTS"]


class GoogleTTS(BaseTTS):
    """Google Text-to-Speech backend using gTTS.

    Free to use, requires internet connection.
    Good quality voices with multi-language support.
    Supports speed control via pydub (requires ffmpeg).

    Install: pip install gTTS pydub
    """

    # Supported languages (subset of most common)
    LANGUAGES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh-CN": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)",
        "ar": "Arabic",
        "hi": "Hindi",
        "nl": "Dutch",
        "pl": "Polish",
        "sv": "Swedish",
        "tr": "Turkish",
        "vi": "Vietnamese",
    }

    def __init__(
        self,
        lang: str = "en",
        slow: bool = False,
        speed: float = 1.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lang = lang
        self.slow = slow
        self.speed = speed  # 1.0 = normal, >1.0 = faster, <1.0 = slower

    @property
    def name(self) -> str:
        return "gtts"

    @property
    def requires_internet(self) -> bool:
        return True

    def synthesize(self, text: str, output_path: str) -> Path:
        """Synthesize text using Google TTS with optional speed control."""
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError(
                "gTTS package not installed. Install with: pip install gTTS"
            )

        # Get language from config or use default
        lang = self.config.get("voice", self.lang)
        if lang in self.LANGUAGES:
            pass  # Valid language code
        elif lang.lower() in [l.lower() for l in self.LANGUAGES.values()]:
            # Convert language name to code
            for code, name in self.LANGUAGES.items():
                if name.lower() == lang.lower():
                    lang = code
                    break

        # Get speed from config or use instance default
        speed = self.config.get("speed", self.speed)

        out_path = Path(output_path)

        if speed != 1.0:
            # Use pydub for speed control
            audio_data = self._synthesize_with_speed(text, lang, speed)
            audio_data.export(str(out_path), format="mp3")
        else:
            # Direct save without speed modification
            tts = gTTS(text=text, lang=lang, slow=self.slow)
            tts.save(str(out_path))

        return out_path

    def _synthesize_with_speed(self, text: str, lang: str, speed: float):
        """Synthesize with speed control using pydub.

        Args:
            text: Text to synthesize.
            lang: Language code.
            speed: Speed multiplier (>1.0 faster, <1.0 slower).

        Returns:
            AudioSegment with adjusted speed.
        """
        try:
            from gtts import gTTS
            from pydub import AudioSegment
        except ImportError as e:
            raise ImportError(
                "pydub package required for speed control. "
                "Install with: pip install pydub"
            ) from e

        # Generate speech to memory buffer
        with io.BytesIO() as buffer:
            gTTS(text=text, lang=lang, slow=self.slow).write_to_fp(buffer)
            buffer.seek(0)
            sound = AudioSegment.from_file(buffer, format="mp3")

        # Apply speed adjustment
        if speed > 1.0:
            # speedup() for faster playback
            sound = sound.speedup(
                playback_speed=speed,
                chunk_size=150,
                crossfade=25
            )
        elif speed < 1.0:
            # For slower playback, adjust frame rate
            new_frame_rate = int(sound.frame_rate * speed)
            sound = sound._spawn(
                sound.raw_data,
                overrides={"frame_rate": new_frame_rate}
            ).set_frame_rate(sound.frame_rate)

        return sound

    def get_voices(self) -> List[dict]:
        """Get available languages as 'voices'."""
        return [
            {"name": name, "id": code, "type": "language"}
            for code, name in self.LANGUAGES.items()
        ]


# EOF
