#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/template/_templates/audio.py
# ----------------------------------------

"""Template for stx.audio text-to-speech module usage."""

TEMPLATE = {
    "name": "Audio Module",
    "description": "stx.audio usage for text-to-speech with multiple backends (pyttsx3, gTTS, ElevenLabs)",
    "filename": "audio_script.py",
    "priority": 5,
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: {timestamp}
# File: {filepath}

"""
stx.audio - Text-to-Speech
==========================

stx.audio provides:
- Multiple TTS backends (pyttsx3, gTTS, ElevenLabs)
- Automatic fallback between backends
- Local and relay playback modes
- Audio generation and caching
- Agent identification for multi-agent workflows

Usage Patterns
--------------
1. MCP: audio_speak, audio_speak_local, audio_speak_relay
2. Python API: stx.audio.speak()
3. Environment: SCITEX_AUDIO_* variables

Backends
--------
- pyttsx3: Offline, fast, robotic voice
- gtts: Google TTS, natural voice, requires internet
- elevenlabs: Premium quality, requires API key
"""

import scitex as stx

# ============================================================
# Pattern 1: Basic TTS (via MCP - Recommended)
# ============================================================

"""
MCP Tools for Claude Code:
--------------------------

# Basic speech (auto-selects backend)
audio_speak(text="Hello, I completed the task")

# With specific backend
audio_speak(text="Processing complete", backend="gtts")

# With agent identification
audio_speak(text="Analysis done", agent_id="analyzer-1")

# Local playback (when MCP server is local)
audio_speak_local(text="Local notification")

# Relay playback (when MCP server is remote, audio plays on client)
audio_speak_relay(text="Remote notification")

# List available backends
audio_list_backends()

# Check audio status
audio_check_audio_status()
"""

# ============================================================
# Pattern 2: Python API Usage
# ============================================================

def python_api_example():
    """Direct Python API for TTS."""
    from scitex.audio import speak, list_backends, check_audio_status

    # Basic speech
    speak("Hello from SciTeX")

    # With options
    speak(
        text="Processing completed successfully",
        backend="gtts",      # or "pyttsx3", "elevenlabs"
        speed=1.5,           # Playback speed multiplier
        wait=True,           # Wait for completion
        save=False,          # Don't save audio file
    )

    # With agent identification (for multi-agent workflows)
    speak(
        text="Agent Alpha reporting",
        agent_id="alpha",
        signature=True       # Add agent signature sound
    )

    # List available backends
    backends = list_backends()
    print(f"Available: {{backends}}")

    # Check audio system status
    status = check_audio_status()
    print(f"Audio status: {{status}}")

# ============================================================
# Pattern 3: Audio Generation (Save to File)
# ============================================================

def generate_audio_example():
    """Generate audio files without playing."""
    from scitex.audio import generate_audio

    # Generate audio file
    result = generate_audio(
        text="This is a test message",
        output_path="notification.mp3",
        backend="gtts"
    )
    print(f"Audio saved to: {{result['path']}}")

    # Generate and get base64 (for embedding)
    result_b64 = generate_audio(
        text="Embedded audio",
        return_base64=True
    )
    # result_b64['base64'] contains the audio data

# ============================================================
# Pattern 4: With @stx.session
# ============================================================

@stx.session
def main(
    message="Analysis complete",
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Session with audio notifications."""
    from scitex.audio import speak

    logger.info("Starting analysis...")

    # Do some work
    import time
    time.sleep(1)

    # Notify via audio when done
    speak(
        text=f"{{message}}. Results saved to {{CONFIG.SDIR_OUT}}",
        speed=1.5,
        wait=True
    )

    logger.info("Done")
    return 0

# ============================================================
# Pattern 5: Environment Configuration
# ============================================================

"""
Environment Variables:
----------------------
SCITEX_AUDIO_BACKEND=gtts           # Default backend
SCITEX_AUDIO_SPEED=1.5              # Default playback speed
SCITEX_AUDIO_VOICE=en               # Voice/language
SCITEX_AUDIO_ELEVENLABS_KEY=xxx     # ElevenLabs API key
SCITEX_AUDIO_RELAY_URL=http://...   # Relay server URL
"""

# ============================================================
# Pattern 6: Multi-Agent Workflows
# ============================================================

"""
Agent Identification:
---------------------
In multi-agent workflows, each agent can identify itself:

# Agent 1 (Architect)
audio_speak(text="Architecture plan ready", agent_id="architect")

# Agent 2 (Developer)
audio_speak(text="Implementation complete", agent_id="developer")

# Agent 3 (Tester)
audio_speak(text="All tests passing", agent_id="tester")

The agent_id helps distinguish which agent is speaking,
especially useful when running concurrent agents.
"""

# ============================================================
# MCP Tools Reference
# ============================================================

"""
MCP Tools:
----------

Speech:
- audio_speak(text, backend, speed, agent_id)
- audio_speak_local(text, ...)     # Play on server
- audio_speak_relay(text, ...)     # Play on client via relay

Generation:
- audio_generate_audio(text, output_path, backend)
- audio_play_audio(path)

Management:
- audio_list_backends()
- audio_list_voices(backend)
- audio_check_audio_status()
- audio_speech_queue_status()
- audio_list_audio_files(limit)
- audio_clear_audio_cache(max_age_hours)

Context:
- audio_announce_context()         # Announce cwd and git branch
"""

if __name__ == "__main__":
    main()
''',
}

__all__ = ["TEMPLATE"]

# EOF
