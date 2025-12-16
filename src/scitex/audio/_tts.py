#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/_tts.py
# ----------------------------------------

"""
Text-to-Speech implementation using ElevenLabs API.

This module provides TTS functionality that can be used:
1. Directly via the ElevenLabs Python SDK
2. Via MCP server integration

Environment Variables:
    ELEVENLABS_API_KEY: Your ElevenLabs API key
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = ["TTS", "speak"]


@dataclass
class TTSConfig:
    """Configuration for TTS."""

    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel (default)
    voice_name: Optional[str] = None
    model_id: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    speed: float = 1.0
    output_format: str = "mp3_44100_128"


class TTS:
    """Text-to-Speech using ElevenLabs API.

    Examples:
        # Basic usage
        tts = TTS()
        tts.speak("Hello, world!")

        # With custom voice
        tts = TTS(voice_name="Rachel")
        tts.speak("Processing complete")

        # Save to file without playing
        tts.speak("Test", output_path="/tmp/test.mp3", play=False)
    """

    # Popular voice presets
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
        voice_name: Optional[str] = None,
        voice_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize TTS.

        Args:
            api_key: ElevenLabs API key. Defaults to ELEVENLABS_API_KEY env var.
            voice_name: Voice name (e.g., "Rachel", "Adam").
            voice_id: Direct voice ID (overrides voice_name).
            **kwargs: Additional config options (stability, speed, etc.)
        """
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.config = TTSConfig(**kwargs)

        if voice_id:
            self.config.voice_id = voice_id
        elif voice_name:
            self.config.voice_name = voice_name
            normalized = voice_name.lower()
            if normalized in self.VOICES:
                self.config.voice_id = self.VOICES[normalized]

        self._client = None

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

    def speak(
        self,
        text: str,
        output_path: Optional[str] = None,
        play: bool = True,
        voice_name: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Convert text to speech and optionally play it.

        Args:
            text: Text to convert to speech.
            output_path: Path to save audio file. Auto-generated if None.
            play: Whether to play the audio after generation.
            voice_name: Override voice name for this call.
            voice_id: Override voice ID for this call.

        Returns:
            Path to the generated audio file, or None if only played.
        """
        # Determine voice
        vid = voice_id or self.config.voice_id
        if voice_name and not voice_id:
            normalized = voice_name.lower()
            vid = self.VOICES.get(normalized, vid)

        # Generate audio
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=vid,
            model_id=self.config.model_id,
            voice_settings={
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
                "style": self.config.style,
                "speed": self.config.speed,
            },
            output_format=self.config.output_format,
        )

        # Determine output path
        if output_path:
            out_path = Path(output_path)
        else:
            suffix = ".mp3" if "mp3" in self.config.output_format else ".wav"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="scitex_tts_")
            os.close(fd)
            out_path = Path(tmp_path)

        # Write audio to file
        with open(out_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        # Play if requested
        if play:
            self._play_audio(out_path)

        return out_path if output_path else None

    def _play_audio(self, path: Path) -> None:
        """Play audio file using available system player.

        Includes Windows fallback for WSL environments.
        """
        # Check if we're in WSL - if so, prefer Windows playback directly
        # to avoid double playback issues with Linux audio hanging
        if os.path.exists("/mnt/c/Windows"):
            if self._play_audio_windows(path):
                return
            # Fall through to Linux players if Windows playback fails

        players = [
            ["mpv", "--no-video", str(path)],
            ["ffplay", "-nodisp", "-autoexit", str(path)],
            ["aplay", str(path)],
            ["afplay", str(path)],  # macOS
        ]

        for player_cmd in players:
            try:
                subprocess.run(
                    player_cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                return
            except subprocess.TimeoutExpired:
                # Audio playback hung, don't try more players
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        print(f"Warning: No audio player found. Audio saved to: {path}")

    def _play_audio_windows(self, path: Path) -> bool:
        """Play audio via Windows PowerShell SoundPlayer (WSL fallback).

        Uses headless SoundPlayer - no GUI popup.
        """
        import shutil
        import tempfile

        # Check if we're in WSL
        if not os.path.exists("/mnt/c/Windows"):
            return False

        powershell = shutil.which("powershell.exe")
        if not powershell:
            return False

        try:
            # SoundPlayer only supports WAV, so convert if needed
            wav_path = path
            if path.suffix.lower() in ('.mp3', '.ogg', '.m4a'):
                try:
                    from pydub import AudioSegment
                    fd, tmp_wav = tempfile.mkstemp(suffix='.wav', prefix='scitex_')
                    os.close(fd)
                    wav_path = Path(tmp_wav)
                    audio = AudioSegment.from_file(str(path))
                    audio.export(str(wav_path), format='wav')
                except ImportError:
                    pass

            result = subprocess.run(
                ["wslpath", "-w", str(wav_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return False

            windows_path = result.stdout.strip()

            ps_command = f'''
$player = New-Object System.Media.SoundPlayer
$player.SoundLocation = "{windows_path}"
$player.PlaySync()
'''
            subprocess.run(
                [powershell, "-NoProfile", "-Command", ps_command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )

            # Clean up temp WAV
            if wav_path != path and wav_path.exists():
                try:
                    wav_path.unlink()
                except Exception:
                    pass

            return True

        except Exception:
            return False

    def list_voices(self) -> list:
        """List available voices from ElevenLabs."""
        response = self.client.voices.get_all()
        return [
            {"name": v.name, "voice_id": v.voice_id, "labels": v.labels}
            for v in response.voices
        ]


# Module-level convenience function
_default_tts: Optional[TTS] = None


def speak(
    text: str,
    voice: Optional[str] = None,
    play: bool = True,
    output_path: Optional[str] = None,
    **kwargs,
) -> Optional[Path]:
    """Convenience function for quick TTS.

    Args:
        text: Text to speak.
        voice: Voice name (e.g., "Rachel", "Adam").
        play: Whether to play audio.
        output_path: Optional path to save audio.
        **kwargs: Additional TTS config options.

    Returns:
        Path to audio file if output_path specified, else None.

    Examples:
        import scitex

        # Simple speak
        scitex.audio.speak("Hello!")

        # With specific voice
        scitex.audio.speak("Processing complete", voice="Adam")

        # Save without playing
        scitex.audio.speak("Test", play=False, output_path="/tmp/test.mp3")
    """
    global _default_tts

    if _default_tts is None or kwargs:
        _default_tts = TTS(**kwargs)

    return _default_tts.speak(
        text=text,
        voice_name=voice,
        play=play,
        output_path=output_path,
    )


# EOF
