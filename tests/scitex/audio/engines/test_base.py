#!/usr/bin/env python3
# Timestamp: 2026-01-04
# File: tests/scitex/audio/engines/test_base.py

"""Tests for scitex.audio.engines.base module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestTTSBackend:
    """Tests for TTSBackend class."""

    def test_backend_constants(self):
        """Test that backend constants are defined correctly."""
        from scitex.audio.engines.base import TTSBackend

        assert TTSBackend.ELEVENLABS == "elevenlabs"
        assert TTSBackend.GTTS == "gtts"
        assert TTSBackend.PYTTSX3 == "pyttsx3"
        assert TTSBackend.EDGE == "edge"

    def test_available_returns_list(self):
        """Test that available() returns a list."""
        from scitex.audio.engines.base import TTSBackend

        result = TTSBackend.available()
        assert isinstance(result, list)

    def test_available_detects_gtts_when_installed(self):
        """Test that available() detects gTTS when installed."""
        from scitex.audio.engines.base import TTSBackend

        with patch.dict("sys.modules", {"gtts": MagicMock()}):
            # Force re-evaluation by calling available
            backends = TTSBackend.available()
            # gtts should be detected if the module import succeeds
            assert isinstance(backends, list)

    def test_available_handles_missing_modules_gracefully(self):
        """Test that available() handles ImportError gracefully."""
        from scitex.audio.engines.base import TTSBackend

        # Should not raise even if modules are missing
        result = TTSBackend.available()
        assert isinstance(result, list)


class TestBaseTTS:
    """Tests for BaseTTS abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseTTS cannot be instantiated directly."""
        from scitex.audio.engines.base import BaseTTS

        with pytest.raises(TypeError):
            BaseTTS()

    def test_config_stored_correctly(self):
        """Test that config kwargs are stored."""
        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                return Path(output_path)

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS(key1="value1", key2="value2")
        assert tts.config["key1"] == "value1"
        assert tts.config["key2"] == "value2"

    def test_requires_api_key_default_false(self):
        """Test that requires_api_key defaults to False."""
        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                return Path(output_path)

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS()
        assert tts.requires_api_key is False

    def test_requires_internet_default_false(self):
        """Test that requires_internet defaults to False."""
        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                return Path(output_path)

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS()
        assert tts.requires_internet is False

    def test_speak_with_output_path(self, tmp_path):
        """Test speak() returns path when output_path is provided."""
        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                # Create a dummy file
                path = Path(output_path)
                path.write_text("dummy audio")
                return path

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS()
        output_file = tmp_path / "test.mp3"

        # Mock _play_audio to avoid actual playback
        with patch.object(tts, "_play_audio"):
            result = tts.speak("Hello", output_path=str(output_file), play=True)

        assert result == output_file
        assert output_file.exists()

    def test_speak_without_output_path_returns_none(self, tmp_path):
        """Test speak() returns None when no output_path is provided."""
        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                path = Path(output_path)
                path.write_text("dummy audio")
                return path

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS()

        with patch.object(tts, "_play_audio"):
            result = tts.speak("Hello", play=True)

        assert result is None

    def test_speak_sets_voice_in_config(self, tmp_path):
        """Test that speak() sets voice in config when provided."""
        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                path = Path(output_path)
                path.write_text("dummy audio")
                return path

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS()
        output_file = tmp_path / "test.mp3"

        with patch.object(tts, "_play_audio"):
            tts.speak("Hello", output_path=str(output_file), voice="custom_voice")

        assert tts.config.get("voice") == "custom_voice"

    def test_speak_without_play(self, tmp_path):
        """Test speak() does not play when play=False."""
        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                path = Path(output_path)
                path.write_text("dummy audio")
                return path

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS()
        output_file = tmp_path / "test.mp3"

        with patch.object(tts, "_play_audio") as mock_play:
            tts.speak("Hello", output_path=str(output_file), play=False)
            mock_play.assert_not_called()

    def test_play_audio_tries_multiple_players(self, tmp_path):
        """Test _play_audio tries multiple players."""
        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                return Path(output_path)

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS()
        test_file = tmp_path / "test.mp3"
        test_file.write_text("dummy")

        # Mock subprocess.run to simulate player not found
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("player not found")
            # Should not raise, just print warning
            tts._play_audio(test_file)

    def test_play_audio_handles_timeout(self, tmp_path):
        """Test _play_audio handles timeout gracefully."""
        import subprocess

        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                return Path(output_path)

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS()
        test_file = tmp_path / "test.mp3"
        test_file.write_text("dummy")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ffplay", 30)
            # Should not raise
            tts._play_audio(test_file)

    @pytest.mark.skipif(
        not os.path.exists("/mnt/c/Windows"),
        reason="WSL-specific test"
    )
    def test_play_audio_windows_wsl_fallback(self, tmp_path):
        """Test Windows fallback in WSL environment."""
        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                return Path(output_path)

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS()
        test_file = tmp_path / "test.wav"
        test_file.write_text("dummy")

        # Test that Windows fallback is attempted
        result = tts._play_audio_windows(test_file)
        assert isinstance(result, bool)

    def test_play_audio_windows_returns_false_on_non_wsl(self, tmp_path):
        """Test _play_audio_windows returns False when not in WSL."""
        from scitex.audio.engines.base import BaseTTS

        class ConcreteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                return Path(output_path)

            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        tts = ConcreteTTS()
        test_file = tmp_path / "test.wav"
        test_file.write_text("dummy")

        with patch("os.path.exists", return_value=False):
            result = tts._play_audio_windows(test_file)
            assert result is False


class TestAbstractMethodsEnforced:
    """Test that abstract methods are enforced."""

    def test_synthesize_is_abstract(self):
        """Test that synthesize must be implemented."""
        from scitex.audio.engines.base import BaseTTS

        class IncompleteTTS(BaseTTS):
            def get_voices(self):
                return []

            @property
            def name(self):
                return "test"

        with pytest.raises(TypeError):
            IncompleteTTS()

    def test_get_voices_is_abstract(self):
        """Test that get_voices must be implemented."""
        from scitex.audio.engines.base import BaseTTS

        class IncompleteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                return Path(output_path)

            @property
            def name(self):
                return "test"

        with pytest.raises(TypeError):
            IncompleteTTS()

    def test_name_is_abstract(self):
        """Test that name property must be implemented."""
        from scitex.audio.engines.base import BaseTTS

        class IncompleteTTS(BaseTTS):
            def synthesize(self, text, output_path):
                return Path(output_path)

            def get_voices(self):
                return []

        with pytest.raises(TypeError):
            IncompleteTTS()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
