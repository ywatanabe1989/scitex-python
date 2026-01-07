#!/usr/bin/env python3
"""Tests for scitex.cli.audio - Text-to-speech CLI commands."""

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.audio import audio


class TestAudioGroup:
    """Tests for the audio command group."""

    def test_audio_help(self):
        """Test that audio help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(audio, ["--help"])
        assert result.exit_code == 0
        assert "Text-to-speech utilities" in result.output

    def test_audio_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(audio, ["--help"])
        expected_commands = ["speak", "backends", "check", "stop"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in audio help"


class TestAudioSpeak:
    """Tests for the audio speak command."""

    def test_speak_basic(self):
        """Test basic speak command."""
        runner = CliRunner()
        with patch("scitex.audio.speak") as mock_speak:
            mock_speak.return_value = None
            result = runner.invoke(audio, ["speak", "Hello world"])
            assert result.exit_code == 0
            mock_speak.assert_called_once()
            call_kwargs = mock_speak.call_args[1]
            assert call_kwargs["text"] == "Hello world"
            assert call_kwargs["play"] is True

    def test_speak_with_backend(self):
        """Test speak command with specific backend."""
        runner = CliRunner()
        with patch("scitex.audio.speak") as mock_speak:
            mock_speak.return_value = None
            result = runner.invoke(audio, ["speak", "Test", "--backend", "gtts"])
            assert result.exit_code == 0
            call_kwargs = mock_speak.call_args[1]
            assert call_kwargs["backend"] == "gtts"

    def test_speak_with_voice(self):
        """Test speak command with voice option."""
        runner = CliRunner()
        with patch("scitex.audio.speak") as mock_speak:
            mock_speak.return_value = None
            result = runner.invoke(audio, ["speak", "Bonjour", "--voice", "fr"])
            assert result.exit_code == 0
            call_kwargs = mock_speak.call_args[1]
            assert call_kwargs["voice"] == "fr"

    def test_speak_with_output(self):
        """Test speak command with output file."""
        runner = CliRunner()
        with patch("scitex.audio.speak") as mock_speak:
            mock_speak.return_value = "/tmp/speech.mp3"
            result = runner.invoke(
                audio, ["speak", "Test", "--output", "/tmp/speech.mp3", "--no-play"]
            )
            assert result.exit_code == 0
            call_kwargs = mock_speak.call_args[1]
            assert call_kwargs["output_path"] == "/tmp/speech.mp3"
            assert call_kwargs["play"] is False

    def test_speak_with_rate(self):
        """Test speak command with rate option (pyttsx3)."""
        runner = CliRunner()
        with patch("scitex.audio.speak") as mock_speak:
            mock_speak.return_value = None
            result = runner.invoke(audio, ["speak", "Fast", "--rate", "200"])
            assert result.exit_code == 0
            call_kwargs = mock_speak.call_args[1]
            assert call_kwargs["rate"] == 200

    def test_speak_with_speed(self):
        """Test speak command with speed option (gtts)."""
        runner = CliRunner()
        with patch("scitex.audio.speak") as mock_speak:
            mock_speak.return_value = None
            result = runner.invoke(audio, ["speak", "Slow", "--speed", "0.8"])
            assert result.exit_code == 0
            call_kwargs = mock_speak.call_args[1]
            assert call_kwargs["speed"] == 0.8

    def test_speak_error_handling(self):
        """Test speak command handles errors gracefully."""
        runner = CliRunner()
        with patch("scitex.audio.speak") as mock_speak:
            mock_speak.side_effect = Exception("TTS failed")
            result = runner.invoke(audio, ["speak", "Test"])
            assert result.exit_code == 1
            assert "Error" in result.output


class TestAudioBackends:
    """Tests for the audio backends command."""

    def test_backends_list(self):
        """Test backends list command."""
        runner = CliRunner()
        with patch("scitex.audio.available_backends") as mock_available:
            with patch(
                "scitex.audio.FALLBACK_ORDER", ["pyttsx3", "gtts", "elevenlabs"]
            ):
                mock_available.return_value = ["pyttsx3", "gtts"]
                result = runner.invoke(audio, ["backends"])
                assert result.exit_code == 0
                assert "Available TTS Backends" in result.output
                assert "pyttsx3" in result.output
                assert "gtts" in result.output

    def test_backends_json(self):
        """Test backends list with --json flag."""
        runner = CliRunner()
        with patch("scitex.audio.available_backends") as mock_available:
            with patch("scitex.audio.FALLBACK_ORDER", ["pyttsx3", "gtts"]):
                mock_available.return_value = ["pyttsx3"]
                result = runner.invoke(audio, ["backends", "--json"])
                assert result.exit_code == 0
                import json

                output = json.loads(result.output)
                assert "available" in output
                assert "fallback_order" in output

    def test_backends_no_available(self):
        """Test backends list when none available."""
        runner = CliRunner()
        with patch("scitex.audio.available_backends") as mock_available:
            with patch("scitex.audio.FALLBACK_ORDER", ["pyttsx3", "gtts"]):
                mock_available.return_value = []
                result = runner.invoke(audio, ["backends"])
                assert result.exit_code == 0
                assert "No backends available" in result.output


class TestAudioCheck:
    """Tests for the audio check command."""

    def test_check_default(self):
        """Test audio check command."""
        runner = CliRunner()
        with patch("scitex.audio.check_wsl_audio") as mock_check:
            mock_check.return_value = {
                "is_wsl": True,
                "wslg_available": True,
                "pulse_server_exists": True,
                "pulse_connected": True,
                "windows_fallback_available": True,
                "recommended": "linux",
            }
            result = runner.invoke(audio, ["check"])
            assert result.exit_code == 0
            assert "Audio Status Check" in result.output
            assert "WSL Environment" in result.output

    def test_check_json(self):
        """Test audio check with --json flag."""
        runner = CliRunner()
        with patch("scitex.audio.check_wsl_audio") as mock_check:
            mock_check.return_value = {
                "is_wsl": False,
                "recommended": "linux",
            }
            result = runner.invoke(audio, ["check", "--json"])
            assert result.exit_code == 0
            import json

            output = json.loads(result.output)
            assert "is_wsl" in output
            assert "recommended" in output

    def test_check_non_wsl(self):
        """Test audio check on non-WSL system."""
        runner = CliRunner()
        with patch("scitex.audio.check_wsl_audio") as mock_check:
            mock_check.return_value = {
                "is_wsl": False,
                "recommended": "linux",
            }
            result = runner.invoke(audio, ["check"])
            assert result.exit_code == 0
            # Should show WSL status but not detailed checks
            assert "WSL Environment" in result.output


class TestAudioStop:
    """Tests for the audio stop command."""

    def test_stop(self):
        """Test audio stop command."""
        runner = CliRunner()
        with patch("scitex.audio.stop_speech") as mock_stop:
            result = runner.invoke(audio, ["stop"])
            assert result.exit_code == 0
            assert "Speech stopped" in result.output
            mock_stop.assert_called_once()

    def test_stop_error(self):
        """Test audio stop handles errors."""
        runner = CliRunner()
        with patch("scitex.audio.stop_speech") as mock_stop:
            mock_stop.side_effect = Exception("Failed to stop")
            result = runner.invoke(audio, ["stop"])
            assert result.exit_code == 1
            assert "Error" in result.output


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
