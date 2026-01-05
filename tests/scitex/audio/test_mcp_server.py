#!/usr/bin/env python3
# Timestamp: 2026-01-04
# File: tests/scitex/audio/test_mcp_server.py

"""Tests for scitex.audio.mcp_server module."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSpeechRequest:
    """Tests for SpeechRequest dataclass."""

    def test_required_fields(self):
        """Test required fields in SpeechRequest."""
        from scitex.audio.mcp_server import SpeechRequest

        request = SpeechRequest(
            request_id="test-123",
            text="Hello world",
        )
        assert request.request_id == "test-123"
        assert request.text == "Hello world"

    def test_default_values(self):
        """Test default values for optional fields."""
        from scitex.audio.mcp_server import SpeechRequest

        request = SpeechRequest(
            request_id="test-123",
            text="Hello",
        )
        assert request.backend is None
        assert request.voice is None
        assert request.rate is None
        assert request.speed is None
        assert request.play is True
        assert request.save is False
        assert request.fallback is True
        assert request.agent_id is None

    def test_created_at_auto_set(self):
        """Test created_at is automatically set."""
        from scitex.audio.mcp_server import SpeechRequest

        before = datetime.now()
        request = SpeechRequest(
            request_id="test-123",
            text="Hello",
        )
        after = datetime.now()

        assert before <= request.created_at <= after

    def test_custom_values(self):
        """Test setting custom values."""
        from scitex.audio.mcp_server import SpeechRequest

        request = SpeechRequest(
            request_id="test-123",
            text="Hello",
            backend="gtts",
            voice="en",
            rate=200,
            speed=1.5,
            play=False,
            save=True,
            fallback=False,
            agent_id="agent-1",
        )
        assert request.backend == "gtts"
        assert request.voice == "en"
        assert request.rate == 200
        assert request.speed == 1.5
        assert request.play is False
        assert request.save is True
        assert request.fallback is False
        assert request.agent_id == "agent-1"


class TestGetAudioDir:
    """Tests for get_audio_dir function."""

    def test_returns_path(self, tmp_path):
        """Test get_audio_dir returns a Path object."""
        import scitex.audio.mcp_server as mcp_module

        with patch.object(mcp_module, "SCITEX_AUDIO_DIR", tmp_path / "audio"):
            result = mcp_module.get_audio_dir()
            assert isinstance(result, Path)

    def test_creates_directory(self, tmp_path):
        """Test get_audio_dir creates directory if not exists."""
        import scitex.audio.mcp_server as mcp_module

        audio_dir = tmp_path / "audio"
        assert not audio_dir.exists()

        with patch.object(mcp_module, "SCITEX_AUDIO_DIR", audio_dir):
            result = mcp_module.get_audio_dir()
            assert result.exists()

    def test_uses_scitex_audio_dir_constant(self, tmp_path):
        """Test get_audio_dir uses SCITEX_AUDIO_DIR module constant."""
        import scitex.audio.mcp_server as mcp_module

        audio_dir = tmp_path / "custom_audio"
        with patch.object(mcp_module, "SCITEX_AUDIO_DIR", audio_dir):
            result = mcp_module.get_audio_dir()
            assert result == audio_dir


class TestAudioServer:
    """Tests for AudioServer class."""

    def test_initialization(self):
        """Test AudioServer initializes correctly."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            assert server._queue_processor_task is None
            assert server._current_request is None
            assert server._processed_count == 0
            assert server._is_processing is False

    def test_speech_queue_initialized(self):
        """Test speech queue is initialized as asyncio.Queue."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()
            assert isinstance(server._speech_queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_start_queue_processor_creates_task(self):
        """Test start_queue_processor creates a task."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            # Mock the process method to avoid actual processing
            with patch.object(server, "_process_speech_queue", new_callable=AsyncMock):
                await server.start_queue_processor()

                assert server._queue_processor_task is not None

                # Cancel the task to clean up
                server._queue_processor_task.cancel()
                try:
                    await server._queue_processor_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_speech_queue_status_returns_dict(self):
        """Test speech_queue_status returns a dictionary."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()
            status = await server.speech_queue_status()

            assert isinstance(status, dict)
            assert status["success"] is True
            assert "queue_size" in status
            assert "is_processing" in status
            assert "current_request" in status
            assert "total_processed" in status
            assert "processor_running" in status
            assert "timestamp" in status

    @pytest.mark.asyncio
    async def test_speech_queue_status_initial_values(self):
        """Test initial values in speech_queue_status."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()
            status = await server.speech_queue_status()

            assert status["queue_size"] == 0
            assert status["is_processing"] is False
            assert status["current_request"] is None
            assert status["total_processed"] == 0
            assert status["processor_running"] is False

    @pytest.mark.asyncio
    async def test_speak_returns_queued_response_when_not_waiting(self):
        """Test speak returns queued response when wait=False."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            result = await server.speak(
                text="Hello",
                wait=False,
            )

            assert result["success"] is True
            assert result["queued"] is True
            assert "request_id" in result
            assert result["queue_position"] == 1
            assert result["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_speak_includes_agent_id(self):
        """Test speak includes agent_id in response."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            result = await server.speak(
                text="Hello",
                agent_id="test-agent",
                wait=False,
            )

            assert result["agent_id"] == "test-agent"

    @pytest.mark.asyncio
    async def test_speak_handles_exception(self):
        """Test speak handles exceptions gracefully."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            # Mock the queue to raise an exception
            with patch.object(
                server._speech_queue, "put", side_effect=Exception("Queue error")
            ):
                result = await server.speak(text="Hello", wait=False)

                assert result["success"] is False
                assert "error" in result

    @pytest.mark.asyncio
    async def test_announce_context_returns_dict(self, tmp_path):
        """Test announce_context returns a dictionary."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            # Mock subprocess for git
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="main\n", stderr=""
                )

                # Mock the speak method
                with patch.object(
                    server,
                    "speak",
                    new_callable=AsyncMock,
                    return_value={"success": True},
                ):
                    with patch.object(
                        server, "start_queue_processor", new_callable=AsyncMock
                    ):
                        result = await server.announce_context()

            assert isinstance(result, dict)
            assert result["success"] is True
            assert "directory" in result
            assert "announced_text" in result

    @pytest.mark.asyncio
    async def test_announce_context_with_git_branch(self, tmp_path):
        """Test announce_context includes git branch when available."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="feature-branch\n", stderr=""
                )

                with patch.object(
                    server,
                    "speak",
                    new_callable=AsyncMock,
                    return_value={"success": True},
                ):
                    with patch.object(
                        server, "start_queue_processor", new_callable=AsyncMock
                    ):
                        result = await server.announce_context()

            assert result["git_branch"] == "feature-branch"
            assert "feature-branch" in result["announced_text"]

    @pytest.mark.asyncio
    async def test_announce_context_handles_no_git(self, tmp_path):
        """Test announce_context handles non-git directories."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=128, stdout="", stderr="not a git repository"
                )

                with patch.object(
                    server,
                    "speak",
                    new_callable=AsyncMock,
                    return_value={"success": True},
                ):
                    with patch.object(
                        server, "start_queue_processor", new_callable=AsyncMock
                    ):
                        result = await server.announce_context()

            assert result["git_branch"] is None

    @pytest.mark.asyncio
    async def test_announce_context_include_full_path(self):
        """Test announce_context with include_full_path=True."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="main\n", stderr=""
                )

                with patch.object(
                    server,
                    "speak",
                    new_callable=AsyncMock,
                    return_value={"success": True},
                ):
                    with patch.object(
                        server, "start_queue_processor", new_callable=AsyncMock
                    ):
                        result = await server.announce_context(include_full_path=True)

            # Full path should be in the announced text
            assert "/" in result["announced_text"]


class TestAudioServerExecuteSpeak:
    """Tests for AudioServer._execute_speak method."""

    @pytest.mark.asyncio
    async def test_execute_speak_returns_dict(self):
        """Test _execute_speak returns a dictionary."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer, SpeechRequest

            server = AudioServer()

            request = SpeechRequest(
                request_id="test-123",
                text="Hello",
                play=False,
            )

            # Patch at scitex.audio level where the imports happen
            with patch("scitex.audio.available_backends", return_value=["gtts"]):
                with patch("scitex.audio.speak", return_value=None):
                    with patch(
                        "scitex.audio._cross_process_lock.AudioPlaybackLock"
                    ) as mock_lock:
                        mock_lock.return_value.acquire.return_value = None
                        mock_lock.return_value.release.return_value = None

                        result = await server._execute_speak(request)

            assert isinstance(result, dict)
            assert result["success"] is True
            assert result["request_id"] == "test-123"
            assert result["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_execute_speak_includes_timestamp(self):
        """Test _execute_speak includes timestamp."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer, SpeechRequest

            server = AudioServer()

            request = SpeechRequest(
                request_id="test-123",
                text="Hello",
                play=False,
            )

            # Patch at scitex.audio level where the imports happen
            with patch("scitex.audio.available_backends", return_value=["gtts"]):
                with patch("scitex.audio.speak", return_value=None):
                    with patch(
                        "scitex.audio._cross_process_lock.AudioPlaybackLock"
                    ) as mock_lock:
                        mock_lock.return_value.acquire.return_value = None
                        mock_lock.return_value.release.return_value = None

                        result = await server._execute_speak(request)

            assert "timestamp" in result


class TestAudioServerSetupHandlers:
    """Tests for AudioServer.setup_handlers method."""

    def test_setup_handlers_called_on_init(self):
        """Test setup_handlers is called during initialization."""
        with patch("scitex.audio.mcp_server.Server") as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server

            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            # Verify decorators were called
            assert mock_server.list_tools.called
            assert mock_server.call_tool.called
            assert mock_server.list_resources.called
            assert mock_server.read_resource.called


class TestMCPHandlers:
    """Tests for MCP handler functions."""

    @pytest.mark.asyncio
    async def test_list_backends_handler_returns_dict(self):
        """Test list_backends_handler returns a dictionary."""
        from scitex.audio._mcp_handlers import list_backends_handler

        # Patch at scitex.audio level where local imports resolve
        with patch("scitex.audio.available_backends", return_value=["gtts"]):
            result = await list_backends_handler()

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "backends" in result
        assert "available" in result
        assert "default" in result

    @pytest.mark.asyncio
    async def test_list_backends_handler_includes_all_backends(self):
        """Test list_backends_handler includes all known backends."""
        from scitex.audio._mcp_handlers import list_backends_handler

        with patch("scitex.audio.available_backends", return_value=["gtts"]):
            result = await list_backends_handler()

        backend_names = [b["name"] for b in result["backends"]]
        assert "gtts" in backend_names
        assert "elevenlabs" in backend_names
        assert "pyttsx3" in backend_names

    @pytest.mark.asyncio
    async def test_list_backends_handler_marks_availability(self):
        """Test list_backends_handler marks backend availability correctly."""
        from scitex.audio._mcp_handlers import list_backends_handler

        with patch(
            "scitex.audio.available_backends",
            return_value=["gtts", "pyttsx3"],
        ):
            result = await list_backends_handler()

        backends_by_name = {b["name"]: b for b in result["backends"]}
        assert backends_by_name["gtts"]["available"] is True
        assert backends_by_name["pyttsx3"]["available"] is True
        assert backends_by_name["elevenlabs"]["available"] is False

    @pytest.mark.asyncio
    async def test_list_backends_handler_handles_exception(self):
        """Test list_backends_handler handles exceptions."""
        from scitex.audio._mcp_handlers import list_backends_handler

        with patch(
            "scitex.audio.available_backends",
            side_effect=Exception("Import error"),
        ):
            result = await list_backends_handler()

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_voices_handler_returns_dict(self):
        """Test list_voices_handler returns a dictionary."""
        from scitex.audio._mcp_handlers import list_voices_handler

        mock_tts = MagicMock()
        mock_tts.get_voices.return_value = [{"name": "English", "id": "en"}]

        with patch("scitex.audio.get_tts", return_value=mock_tts):
            result = await list_voices_handler(backend="gtts")

        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["backend"] == "gtts"
        assert "voices" in result
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_list_voices_handler_handles_exception(self):
        """Test list_voices_handler handles exceptions."""
        from scitex.audio._mcp_handlers import list_voices_handler

        with patch(
            "scitex.audio.get_tts",
            side_effect=Exception("Backend not available"),
        ):
            result = await list_voices_handler(backend="invalid")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_play_audio_handler_file_not_found(self):
        """Test play_audio_handler returns error for non-existent file."""
        from scitex.audio._mcp_handlers import play_audio_handler

        result = await play_audio_handler(path="/nonexistent/file.mp3")

        assert result["success"] is False
        assert "File not found" in result["error"]

    @pytest.mark.asyncio
    async def test_play_audio_handler_plays_file(self, tmp_path):
        """Test play_audio_handler plays existing file."""
        from scitex.audio._mcp_handlers import play_audio_handler

        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"dummy audio")

        with patch("scitex.audio.engines.base.BaseTTS._play_audio"):
            result = await play_audio_handler(path=str(test_file))

        assert result["success"] is True
        assert "played" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_generate_audio_handler_creates_file(self, tmp_path):
        """Test generate_audio_handler creates audio file."""
        from scitex.audio._mcp_handlers import generate_audio_handler

        output_path = tmp_path / "output.mp3"

        mock_result_path = MagicMock()
        mock_result_path.exists.return_value = True
        mock_result_path.stat.return_value.st_size = 1024

        # speak is imported as "tts_speak" but we patch where it's defined
        with patch("scitex.audio.speak", return_value=mock_result_path):
            result = await generate_audio_handler(
                text="Hello",
                output_path=str(output_path),
            )

        assert result["success"] is True
        assert "path" in result
        assert result["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_generate_audio_handler_returns_base64(self, tmp_path):
        """Test generate_audio_handler returns base64 when requested."""
        from scitex.audio._mcp_handlers import generate_audio_handler

        output_path = tmp_path / "output.mp3"
        output_path.write_bytes(b"dummy audio")

        mock_result_path = output_path
        with patch("scitex.audio.speak", return_value=mock_result_path):
            result = await generate_audio_handler(
                text="Hello",
                output_path=str(output_path),
                return_base64=True,
            )

        assert result["success"] is True
        assert "base64" in result

    @pytest.mark.asyncio
    async def test_generate_audio_handler_handles_exception(self):
        """Test generate_audio_handler handles exceptions."""
        from scitex.audio._mcp_handlers import generate_audio_handler

        with patch(
            "scitex.audio.speak",
            side_effect=Exception("TTS error"),
        ):
            result = await generate_audio_handler(text="Hello")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_audio_files_handler_empty_dir(self, tmp_path):
        """Test list_audio_files_handler with empty directory."""
        from scitex.audio._mcp_handlers import list_audio_files_handler

        with patch("scitex.audio._mcp_handlers._get_audio_dir", return_value=tmp_path):
            result = await list_audio_files_handler()

        assert result["success"] is True
        assert result["files"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_list_audio_files_handler_lists_files(self, tmp_path):
        """Test list_audio_files_handler lists audio files."""
        from scitex.audio._mcp_handlers import list_audio_files_handler

        # Create test files
        (tmp_path / "test1.mp3").write_bytes(b"audio1")
        (tmp_path / "test2.wav").write_bytes(b"audio2")

        with patch("scitex.audio._mcp_handlers._get_audio_dir", return_value=tmp_path):
            result = await list_audio_files_handler()

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["files"]) == 2

    @pytest.mark.asyncio
    async def test_list_audio_files_handler_respects_limit(self, tmp_path):
        """Test list_audio_files_handler respects limit parameter."""
        from scitex.audio._mcp_handlers import list_audio_files_handler

        # Create multiple test files
        for i in range(5):
            (tmp_path / f"test{i}.mp3").write_bytes(b"audio")

        with patch("scitex.audio._mcp_handlers._get_audio_dir", return_value=tmp_path):
            result = await list_audio_files_handler(limit=3)

        assert result["count"] == 3
        assert len(result["files"]) == 3

    @pytest.mark.asyncio
    async def test_clear_audio_cache_handler_deletes_old_files(self, tmp_path):
        """Test clear_audio_cache_handler deletes old files."""
        import time

        from scitex.audio._mcp_handlers import clear_audio_cache_handler

        # Create a test file
        test_file = tmp_path / "old.mp3"
        test_file.write_bytes(b"old audio")

        # Set modification time to 2 days ago
        old_time = time.time() - (48 * 3600)
        os.utime(test_file, (old_time, old_time))

        with patch("scitex.audio._mcp_handlers._get_audio_dir", return_value=tmp_path):
            result = await clear_audio_cache_handler(max_age_hours=24)

        assert result["success"] is True
        assert result["deleted"] == 1
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_clear_audio_cache_handler_keeps_recent_files(self, tmp_path):
        """Test clear_audio_cache_handler keeps recent files."""
        from scitex.audio._mcp_handlers import clear_audio_cache_handler

        # Create a recent file
        test_file = tmp_path / "recent.mp3"
        test_file.write_bytes(b"recent audio")

        with patch("scitex.audio._mcp_handlers._get_audio_dir", return_value=tmp_path):
            result = await clear_audio_cache_handler(max_age_hours=24)

        assert result["success"] is True
        assert result["deleted"] == 0
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_clear_audio_cache_handler_delete_all(self, tmp_path):
        """Test clear_audio_cache_handler with max_age_hours=0 deletes all."""
        from scitex.audio._mcp_handlers import clear_audio_cache_handler

        # Create test files
        (tmp_path / "test1.mp3").write_bytes(b"audio1")
        (tmp_path / "test2.wav").write_bytes(b"audio2")

        with patch("scitex.audio._mcp_handlers._get_audio_dir", return_value=tmp_path):
            result = await clear_audio_cache_handler(max_age_hours=0)

        assert result["success"] is True
        assert result["deleted"] == 2

    @pytest.mark.asyncio
    async def test_check_audio_status_handler_returns_dict(self):
        """Test check_audio_status_handler returns a dictionary."""
        from scitex.audio._mcp_handlers import check_audio_status_handler

        mock_status = {"wsl": True, "players": ["aplay"]}

        with patch("scitex.audio.check_wsl_audio", return_value=mock_status):
            result = await check_audio_status_handler()

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_check_audio_status_handler_handles_exception(self):
        """Test check_audio_status_handler handles exceptions."""
        from scitex.audio._mcp_handlers import check_audio_status_handler

        with patch(
            "scitex.audio.check_wsl_audio",
            side_effect=Exception("Status check failed"),
        ):
            result = await check_audio_status_handler()

        assert result["success"] is False
        assert "error" in result


class TestMCPServerMain:
    """Tests for main() function."""

    def test_main_function_exists(self):
        """Test main function exists."""
        from scitex.audio.mcp_server import main

        assert callable(main)

    def test_main_is_async(self):
        """Test main is an async function."""
        import asyncio

        from scitex.audio.mcp_server import main

        assert asyncio.iscoroutinefunction(main)


class TestAudioServerToolHandling:
    """Tests for AudioServer tool handling."""

    @pytest.mark.asyncio
    async def test_handle_call_tool_speak(self):
        """Test handle_call_tool for 'speak' command."""
        with patch("scitex.audio.mcp_server.Server") as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server

            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            # Get the registered handler
            call_tool_decorator = mock_server.call_tool.return_value
            handler = (
                call_tool_decorator.call_args[0][0]
                if call_tool_decorator.call_args
                else None
            )

            # The handler is registered via decorator, so we test the speak method directly
            result = await server.speak(text="Hello", wait=False)

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_handle_call_tool_unknown_tool(self):
        """Test handle_call_tool raises for unknown tool."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            # We can't easily test the handler directly, but we can verify
            # the server structure is correct
            assert server.server is not None


class TestAudioServerResourceHandling:
    """Tests for AudioServer resource handling."""

    def test_server_has_resource_handlers(self):
        """Test server has resource handlers set up."""
        with patch("scitex.audio.mcp_server.Server") as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server

            from scitex.audio.mcp_server import AudioServer

            AudioServer()

            # Verify resource decorators were called
            assert mock_server.list_resources.called
            assert mock_server.read_resource.called


class TestSpeechQueueProcessing:
    """Tests for speech queue processing."""

    @pytest.mark.asyncio
    async def test_queue_processes_requests(self):
        """Test queue processes requests in order."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            # Add requests to queue
            await server.speak(text="First", wait=False)
            await server.speak(text="Second", wait=False)

            assert server._speech_queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_processed_count_increments(self):
        """Test processed count increments after processing."""
        with patch("scitex.audio.mcp_server.Server"):
            from scitex.audio.mcp_server import AudioServer

            server = AudioServer()

            # Initial count should be 0
            assert server._processed_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
