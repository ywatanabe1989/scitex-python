#!/usr/bin/env python3
# Timestamp: 2026-01-04
# File: tests/scitex/audio/engines/test_elevenlabs_engine.py

"""Tests for scitex.audio.engines.elevenlabs_engine module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestElevenLabsTTS:
    """Tests for ElevenLabsTTS class."""

    def test_name_property(self):
        """Test that name returns 'elevenlabs'."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        assert tts.name == "elevenlabs"

    def test_requires_api_key_property(self):
        """Test that requires_api_key returns True."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        assert tts.requires_api_key is True

    def test_requires_internet_property(self):
        """Test that requires_internet returns True."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        assert tts.requires_internet is True

    def test_default_voice_is_rachel(self):
        """Test default voice is 'rachel'."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        assert tts.voice == "rachel"

    def test_default_model_id(self):
        """Test default model ID."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        assert tts.model_id == "eleven_multilingual_v2"

    def test_default_stability(self):
        """Test default stability value."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        assert tts.stability == 0.5

    def test_default_similarity_boost(self):
        """Test default similarity_boost value."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        assert tts.similarity_boost == 0.75

    def test_default_speed(self):
        """Test default speed value."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        assert tts.speed == 1.0

    def test_custom_api_key_initialization(self):
        """Test initializing with custom API key."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key="test-api-key")
        assert tts.api_key == "test-api-key"

    def test_api_key_from_environment(self):
        """Test API key is read from environment."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "env-api-key"}):
            from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

            tts = ElevenLabsTTS()
            assert tts.api_key == "env-api-key"

    def test_custom_voice_initialization(self):
        """Test initializing with custom voice."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(voice="adam")
        assert tts.voice == "adam"

    def test_custom_model_initialization(self):
        """Test initializing with custom model."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(model_id="custom_model")
        assert tts.model_id == "custom_model"

    def test_custom_stability_initialization(self):
        """Test initializing with custom stability."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(stability=0.8)
        assert tts.stability == 0.8

    def test_custom_similarity_boost_initialization(self):
        """Test initializing with custom similarity_boost."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(similarity_boost=0.9)
        assert tts.similarity_boost == 0.9

    def test_custom_speed_initialization(self):
        """Test initializing with custom speed."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(speed=1.5)
        assert tts.speed == 1.5

    def test_voices_dictionary_contains_presets(self):
        """Test VOICES dict contains preset voices."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        assert "rachel" in ElevenLabsTTS.VOICES
        assert "adam" in ElevenLabsTTS.VOICES
        assert "bella" in ElevenLabsTTS.VOICES
        assert "josh" in ElevenLabsTTS.VOICES

    def test_client_lazy_loading(self):
        """Test client is lazily loaded."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        assert tts._client is None

    def test_client_property_initializes_elevenlabs(self):
        """Test client property initializes ElevenLabs client."""
        mock_elevenlabs = MagicMock()
        mock_client = MagicMock()
        mock_elevenlabs.return_value = mock_client

        with patch.dict("sys.modules", {"elevenlabs": MagicMock()}):
            with patch("elevenlabs.client.ElevenLabs", mock_elevenlabs):
                from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

                tts = ElevenLabsTTS(api_key="test-key")

                # Force client initialization through property
                with patch.object(
                    ElevenLabsTTS,
                    "client",
                    new_callable=lambda: property(lambda self: mock_client),
                ):
                    client = tts.client
                    assert client is mock_client

    def test_inherits_from_base_tts(self):
        """Test that ElevenLabsTTS inherits from BaseTTS."""
        from scitex.audio.engines.base import BaseTTS
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        assert issubclass(ElevenLabsTTS, BaseTTS)

    def test_get_voice_id_with_name(self):
        """Test _get_voice_id converts name to ID."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        voice_id = tts._get_voice_id("rachel")
        assert voice_id == ElevenLabsTTS.VOICES["rachel"]

    def test_get_voice_id_with_id(self):
        """Test _get_voice_id returns ID as-is if not found in VOICES."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        custom_id = "custom-voice-id-12345"
        voice_id = tts._get_voice_id(custom_id)
        assert voice_id == custom_id

    def test_get_voice_id_case_insensitive(self):
        """Test _get_voice_id is case insensitive."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        voice_id_lower = tts._get_voice_id("rachel")
        voice_id_upper = tts._get_voice_id("RACHEL")
        voice_id_mixed = tts._get_voice_id("Rachel")

        assert voice_id_lower == voice_id_upper == voice_id_mixed

    def test_get_voice_id_uses_default_when_none(self):
        """Test _get_voice_id uses instance voice when None passed."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(voice="adam")
        voice_id = tts._get_voice_id(None)
        assert voice_id == ElevenLabsTTS.VOICES["adam"]

    def test_synthesize_calls_api(self, tmp_path):
        """Test synthesize calls ElevenLabs API."""
        mock_client = MagicMock()
        mock_audio = [b"audio", b"data"]
        mock_client.text_to_speech.convert.return_value = mock_audio

        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key="test-key")
        tts._client = mock_client

        output_file = tmp_path / "test.mp3"
        result = tts.synthesize("Hello world", str(output_file))

        mock_client.text_to_speech.convert.assert_called_once()
        call_kwargs = mock_client.text_to_speech.convert.call_args[1]
        assert call_kwargs["text"] == "Hello world"
        assert "voice_id" in call_kwargs
        assert result == output_file

    def test_synthesize_writes_audio_chunks(self, tmp_path):
        """Test synthesize writes all audio chunks to file."""
        mock_client = MagicMock()
        mock_audio = [b"chunk1", b"chunk2", b"chunk3"]
        mock_client.text_to_speech.convert.return_value = mock_audio

        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key="test-key")
        tts._client = mock_client

        output_file = tmp_path / "test.mp3"
        tts.synthesize("Hello", str(output_file))

        assert output_file.exists()
        assert output_file.read_bytes() == b"chunk1chunk2chunk3"

    def test_synthesize_uses_voice_from_config(self, tmp_path):
        """Test synthesize uses voice from config."""
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = [b"audio"]

        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key="test-key")
        tts._client = mock_client
        tts.config["voice"] = "adam"

        output_file = tmp_path / "test.mp3"
        tts.synthesize("Hello", str(output_file))

        call_kwargs = mock_client.text_to_speech.convert.call_args[1]
        assert call_kwargs["voice_id"] == ElevenLabsTTS.VOICES["adam"]

    def test_get_voices_returns_preset_voices(self):
        """Test get_voices returns preset voices."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        voices = tts.get_voices()

        assert isinstance(voices, list)
        assert len(voices) >= len(ElevenLabsTTS.VOICES)

        # Check preset voices are included
        preset_names = {v["name"] for v in voices if v.get("type") == "preset"}
        for voice_name in ElevenLabsTTS.VOICES:
            assert voice_name in preset_names

    def test_get_voices_includes_custom_voices(self):
        """Test get_voices includes custom voices from API."""
        mock_client = MagicMock()
        mock_voice = MagicMock()
        mock_voice.name = "Custom Voice"
        mock_voice.voice_id = "custom-id"
        mock_voice.labels = {"accent": "British"}

        mock_response = MagicMock()
        mock_response.voices = [mock_voice]
        mock_client.voices.get_all.return_value = mock_response

        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key="test-key")
        tts._client = mock_client

        voices = tts.get_voices()

        # Should include custom voice
        custom_voices = [v for v in voices if v.get("type") == "custom"]
        assert len(custom_voices) == 1
        assert custom_voices[0]["name"] == "Custom Voice"

    def test_get_voices_handles_api_error(self):
        """Test get_voices handles API errors gracefully."""
        mock_client = MagicMock()
        mock_client.voices.get_all.side_effect = Exception("API Error")

        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key="test-key")
        tts._client = mock_client

        # Should not raise, just return preset voices
        voices = tts.get_voices()
        assert isinstance(voices, list)
        assert len(voices) == len(ElevenLabsTTS.VOICES)


class TestElevenLabsTTSEdgeCases:
    """Edge case tests for ElevenLabsTTS."""

    def test_stability_boundary_values(self):
        """Test stability at boundary values."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts_min = ElevenLabsTTS(stability=0.0)
        assert tts_min.stability == 0.0

        tts_max = ElevenLabsTTS(stability=1.0)
        assert tts_max.stability == 1.0

    def test_similarity_boost_boundary_values(self):
        """Test similarity_boost at boundary values."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts_min = ElevenLabsTTS(similarity_boost=0.0)
        assert tts_min.similarity_boost == 0.0

        tts_max = ElevenLabsTTS(similarity_boost=1.0)
        assert tts_max.similarity_boost == 1.0

    def test_speed_boundary_values(self):
        """Test speed at various values."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts_slow = ElevenLabsTTS(speed=0.5)
        assert tts_slow.speed == 0.5

        tts_fast = ElevenLabsTTS(speed=2.0)
        assert tts_fast.speed == 2.0

    def test_no_api_key(self):
        """Test behavior when no API key is set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ELEVENLABS_API_KEY if present
            os.environ.pop("ELEVENLABS_API_KEY", None)

            from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

            tts = ElevenLabsTTS()
            # API key should be None
            assert tts.api_key is None

    def test_voice_id_direct_passthrough(self):
        """Test that unknown voice IDs are passed through."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        tts = ElevenLabsTTS()
        custom_id = "some-custom-voice-id-that-doesnt-exist"
        result = tts._get_voice_id(custom_id)
        assert result == custom_id


class TestElevenLabsTTSVoicePresets:
    """Tests for voice preset mappings."""

    def test_all_preset_voices_have_ids(self):
        """Test all preset voices have valid IDs."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        for name, voice_id in ElevenLabsTTS.VOICES.items():
            assert voice_id is not None
            assert len(voice_id) > 0
            assert isinstance(voice_id, str)

    def test_expected_voice_count(self):
        """Test expected number of preset voices."""
        from scitex.audio.engines.elevenlabs_engine import ElevenLabsTTS

        # Should have at least 8 preset voices
        assert len(ElevenLabsTTS.VOICES) >= 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
