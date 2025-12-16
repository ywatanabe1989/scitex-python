#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/__init__.py
# ----------------------------------------

"""
SciTeX Audio Module - Text-to-Speech with Multiple Backends

Fallback order: pyttsx3 -> gtts -> elevenlabs

Backends:
    - pyttsx3: System TTS (offline, free, uses espeak/SAPI5)
    - gtts: Google TTS (free, requires internet)
    - elevenlabs: ElevenLabs (paid, high quality)

Usage:
    import scitex

    # Auto-select with fallback (pyttsx3 -> gtts -> elevenlabs)
    scitex.audio.speak("Hello, world!")

    # Specify backend
    scitex.audio.speak("Hello", backend="gtts")

    # Use TTS class directly
    from scitex.audio import GoogleTTS, ElevenLabsTTS, SystemTTS
    tts = SystemTTS()
    tts.speak("Hello!")
"""

import subprocess
from typing import List, Optional

# Import from engines subpackage
from .engines import (
    BaseTTS,
    TTSBackend,
    SystemTTS,
    GoogleTTS,
    ElevenLabsTTS,
)


def stop_speech() -> None:
    """Stop any currently playing speech by killing espeak processes."""
    try:
        subprocess.run(["pkill", "-9", "espeak"], capture_output=True)
    except Exception:
        pass


def check_wsl_audio() -> dict:
    """Check WSL audio status and connectivity.

    Returns:
        dict with keys:
        - is_wsl: bool - whether running in WSL
        - wslg_available: bool - whether WSLg is available
        - pulse_server_exists: bool - whether PulseServer socket exists
        - pulse_connected: bool - whether PulseAudio connection works
        - windows_fallback_available: bool - whether Windows fallback is available
        - recommended: str - recommended playback method
    """
    import os
    import shutil

    result = {
        "is_wsl": False,
        "wslg_available": False,
        "pulse_server_exists": False,
        "pulse_connected": False,
        "windows_fallback_available": False,
        "recommended": "linux",
    }

    # Check if in WSL
    if os.path.exists("/mnt/c/Windows"):
        result["is_wsl"] = True

        # Check WSLg
        if os.path.exists("/mnt/wslg"):
            result["wslg_available"] = True

        # Check PulseServer socket
        if os.path.exists("/mnt/wslg/PulseServer"):
            result["pulse_server_exists"] = True

            # Try to connect to PulseAudio
            try:
                env = os.environ.copy()
                env["PULSE_SERVER"] = "unix:/mnt/wslg/PulseServer"
                proc = subprocess.run(
                    ["pactl", "info"],
                    capture_output=True,
                    timeout=5,
                    env=env,
                )
                if proc.returncode == 0:
                    result["pulse_connected"] = True
            except Exception:
                pass

        # Check Windows fallback
        if shutil.which("powershell.exe"):
            result["windows_fallback_available"] = True

        # Determine recommendation
        if result["pulse_connected"]:
            result["recommended"] = "linux"
        elif result["windows_fallback_available"]:
            result["recommended"] = "windows"
        else:
            result["recommended"] = "none"
    else:
        # Native Linux
        result["recommended"] = "linux"

    return result

# Keep legacy TTS import for backwards compatibility
from ._tts import TTS

__all__ = [
    "speak",
    "stop_speech",
    "check_wsl_audio",
    "TTS",
    "GoogleTTS",
    "ElevenLabsTTS",
    "SystemTTS",
    "BaseTTS",
    "TTSBackend",
    "get_tts",
    "available_backends",
    "start_mcp_server",
    "FALLBACK_ORDER",
]

# Fallback order: pyttsx3 (offline, free) -> gtts (free) -> elevenlabs (paid)
# FALLBACK_ORDER = ["pyttsx3", "gtts", "elevenlabs"]
FALLBACK_ORDER = ["gtts", "pyttsx3", "elevenlabs"]


def available_backends() -> List[str]:
    """Return list of available TTS backends in fallback order."""
    backends = []

    # Check pyttsx3 (offline)
    if SystemTTS:
        try:
            import pyttsx3
            # Try to init to check if espeak is available
            engine = pyttsx3.init()
            engine.stop()
            backends.append("pyttsx3")
        except Exception:
            pass

    # Check gTTS (requires internet, but no API key)
    if GoogleTTS:
        backends.append("gtts")

    # Check ElevenLabs (requires API key)
    if ElevenLabsTTS:
        import os
        if os.environ.get("ELEVENLABS_API_KEY"):
            backends.append("elevenlabs")

    return backends


def get_tts(backend: Optional[str] = None, **kwargs) -> BaseTTS:
    """Get a TTS instance for the specified backend.

    Args:
        backend: Backend name ('pyttsx3', 'gtts', 'elevenlabs').
                 Auto-selects with fallback if None.
        **kwargs: Backend-specific options.

    Returns:
        TTS instance.

    Raises:
        ValueError: If no backends available or backend not found.
    """
    backends = available_backends()

    if not backends:
        raise ValueError(
            "No TTS backends available. Install one of:\n"
            "  pip install pyttsx3       # System TTS (offline, free)\n"
            "    + Linux: sudo apt install espeak-ng libespeak1\n"
            "  pip install gTTS          # Google TTS (free, needs internet)\n"
            "  pip install elevenlabs    # ElevenLabs (paid, best quality)"
        )

    if backend is None:
        # Use fallback order: pyttsx3 -> gtts -> elevenlabs
        for b in FALLBACK_ORDER:
            if b in backends:
                backend = b
                break

    if backend == "pyttsx3" and SystemTTS and "pyttsx3" in backends:
        return SystemTTS(**kwargs)
    elif backend == "gtts" and GoogleTTS:
        return GoogleTTS(**kwargs)
    elif backend == "elevenlabs" and ElevenLabsTTS:
        return ElevenLabsTTS(**kwargs)
    else:
        raise ValueError(
            f"Backend '{backend}' not available. "
            f"Available: {backends}"
        )


def _try_speak_with_fallback(
    text: str,
    voice: Optional[str] = None,
    play: bool = True,
    output_path: Optional[str] = None,
    **kwargs,
) -> tuple:
    """Try to speak with fallback through backends.

    Returns:
        (result, backend_used, error_log)
    """
    backends = available_backends()
    errors = []

    for backend in FALLBACK_ORDER:
        if backend not in backends:
            continue

        try:
            tts = get_tts(backend, **kwargs)
            result = tts.speak(
                text=text,
                voice=voice,
                play=play,
                output_path=output_path,
            )
            return (result, backend, errors)
        except Exception as e:
            errors.append(f"{backend}: {str(e)}")
            continue

    return (None, None, errors)


# Cache for default TTS instance
_default_tts: Optional[BaseTTS] = None
_default_backend: Optional[str] = None


def speak(
    text: str,
    backend: Optional[str] = None,
    voice: Optional[str] = None,
    play: bool = True,
    output_path: Optional[str] = None,
    fallback: bool = True,
    rate: Optional[int] = None,
    speed: Optional[float] = None,
    **kwargs,
) -> Optional[str]:
    """Convert text to speech with automatic fallback.

    Fallback order: pyttsx3 -> gtts -> elevenlabs

    Args:
        text: Text to speak.
        backend: TTS backend ('pyttsx3', 'gtts', 'elevenlabs').
                 Auto-selects with fallback if None.
        voice: Voice name, ID, or language code.
        play: Whether to play the audio.
        output_path: Path to save audio file.
        fallback: If True, try next backend on failure.
        rate: Speech rate in words per minute (pyttsx3 only, default 150).
        speed: Speed multiplier for gtts (1.0=normal, >1.0=faster, <1.0=slower).
        **kwargs: Additional backend options.

    Returns:
        Path to audio file if output_path specified, else None.

    Examples:
        import scitex

        # Simple (auto-selects with fallback)
        scitex.audio.speak("Hello!")

        # Faster speech (pyttsx3)
        scitex.audio.speak("Hello", rate=200)

        # Faster speech (gtts with pydub)
        scitex.audio.speak("Hello", backend="gtts", speed=1.5)

        # Specific backend (no fallback)
        scitex.audio.speak("Hello", backend="pyttsx3", fallback=False)

        # Different language (gTTS)
        scitex.audio.speak("Bonjour", backend="gtts", voice="fr")

        # Save to file
        scitex.audio.speak("Test", output_path="/tmp/test.mp3")
    """
    global _default_tts, _default_backend

    # Stop any previously running speech first
    stop_speech()

    # Pass rate to kwargs for pyttsx3
    if rate is not None:
        kwargs["rate"] = rate

    # Pass speed to kwargs for gtts
    if speed is not None:
        kwargs["speed"] = speed

    # If specific backend requested without fallback
    if backend and not fallback:
        tts = get_tts(backend, **kwargs)
        result = tts.speak(
            text=text,
            voice=voice,
            play=play,
            output_path=output_path,
        )
        return str(result) if result else None

    # Use fallback logic
    if fallback and backend is None:
        result, used_backend, errors = _try_speak_with_fallback(
            text=text,
            voice=voice,
            play=play,
            output_path=output_path,
            **kwargs,
        )
        if result is None and errors:
            raise RuntimeError(
                f"All TTS backends failed:\n" + "\n".join(errors)
            )
        return str(result) if result else None

    # Specific backend with fallback enabled
    try:
        tts = get_tts(backend, **kwargs)
        result = tts.speak(
            text=text,
            voice=voice,
            play=play,
            output_path=output_path,
        )
        return str(result) if result else None
    except Exception as e:
        if fallback:
            # Try other backends
            result, used_backend, errors = _try_speak_with_fallback(
                text=text,
                voice=voice,
                play=play,
                output_path=output_path,
                **kwargs,
            )
            if result is None:
                raise RuntimeError(
                    f"Primary backend '{backend}' failed: {e}\n"
                    f"Fallback errors:\n" + "\n".join(errors)
                )
            return str(result) if result else None
        raise


def start_mcp_server():
    """Start the MCP server for audio."""
    import asyncio
    from .mcp_server import main
    asyncio.run(main())


# EOF
