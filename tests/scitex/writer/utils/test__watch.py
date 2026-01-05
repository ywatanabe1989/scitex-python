#!/usr/bin/env python3
"""Tests for scitex.writer.utils._watch."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from scitex.writer.utils._watch import watch_manuscript


class TestWatchManuscriptCompileScript:
    """Tests for watch_manuscript compile script handling."""

    def test_returns_early_when_compile_script_missing(self, tmp_path):
        """Verify returns immediately when compile script doesn't exist."""
        with patch("subprocess.Popen") as mock_popen:
            watch_manuscript(tmp_path)

            mock_popen.assert_not_called()

    def test_creates_correct_command(self, tmp_path):
        """Verify correct command is built for watch mode."""
        compile_script = tmp_path / "compile"
        compile_script.write_text("#!/bin/bash\necho 'compiling'")

        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = ""  # Stop iteration immediately

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            watch_manuscript(tmp_path)

            expected_cmd = [str(compile_script), "-m", "-w"]
            mock_popen.assert_called_once()
            actual_cmd = mock_popen.call_args[0][0]
            assert actual_cmd == expected_cmd


class TestWatchManuscriptProcess:
    """Tests for watch_manuscript process handling."""

    def test_runs_process_with_correct_settings(self, tmp_path):
        """Verify Popen is called with correct settings."""
        compile_script = tmp_path / "compile"
        compile_script.write_text("#!/bin/bash\necho 'compiling'")

        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = ""

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            watch_manuscript(tmp_path)

            call_kwargs = mock_popen.call_args[1]
            assert call_kwargs["cwd"] == tmp_path
            assert call_kwargs["stdout"] == subprocess.PIPE
            assert call_kwargs["stderr"] == subprocess.STDOUT
            assert call_kwargs["text"] is True
            assert call_kwargs["bufsize"] == 1

    def test_waits_for_process_completion(self, tmp_path):
        """Verify process.wait is called with timeout."""
        compile_script = tmp_path / "compile"
        compile_script.write_text("#!/bin/bash\necho 'compiling'")

        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = ""

        with patch("subprocess.Popen", return_value=mock_process):
            watch_manuscript(tmp_path, timeout=30)

            mock_process.wait.assert_called_once_with(timeout=30)


class TestWatchManuscriptCallback:
    """Tests for watch_manuscript callback handling."""

    def test_calls_callback_on_compilation_event(self, tmp_path):
        """Verify callback is called when 'Compilation' appears in output."""
        compile_script = tmp_path / "compile"
        compile_script.write_text("#!/bin/bash\necho 'compiling'")

        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = [
            "Starting...\n",
            "Compilation complete\n",
            "",  # End iteration
        ]

        callback = MagicMock()

        with patch("subprocess.Popen", return_value=mock_process):
            watch_manuscript(tmp_path, on_compile=callback)

            callback.assert_called_once()

    def test_callback_not_called_without_compilation_keyword(self, tmp_path):
        """Verify callback is not called for non-compilation output."""
        compile_script = tmp_path / "compile"
        compile_script.write_text("#!/bin/bash\necho 'compiling'")

        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = [
            "Starting...\n",
            "Processing files...\n",
            "",
        ]

        callback = MagicMock()

        with patch("subprocess.Popen", return_value=mock_process):
            watch_manuscript(tmp_path, on_compile=callback)

            callback.assert_not_called()

    def test_callback_error_does_not_stop_watch(self, tmp_path):
        """Verify callback errors are caught and logged."""
        compile_script = tmp_path / "compile"
        compile_script.write_text("#!/bin/bash\necho 'compiling'")

        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = [
            "Compilation complete\n",
            "",
        ]

        callback = MagicMock(side_effect=Exception("Callback failed"))

        with patch("subprocess.Popen", return_value=mock_process):
            # Should not raise
            watch_manuscript(tmp_path, on_compile=callback)


class TestWatchManuscriptExceptions:
    """Tests for watch_manuscript exception handling."""

    def test_keyboard_interrupt_terminates_process(self, tmp_path):
        """Verify KeyboardInterrupt terminates the process."""
        compile_script = tmp_path / "compile"
        compile_script.write_text("#!/bin/bash\necho 'compiling'")

        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = KeyboardInterrupt()

        with patch("subprocess.Popen", return_value=mock_process):
            watch_manuscript(tmp_path)

            mock_process.terminate.assert_called_once()

    def test_generic_exception_terminates_process(self, tmp_path):
        """Verify generic exceptions terminate the process."""
        compile_script = tmp_path / "compile"
        compile_script.write_text("#!/bin/bash\necho 'compiling'")

        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = RuntimeError("Connection lost")

        with patch("subprocess.Popen", return_value=mock_process):
            watch_manuscript(tmp_path)

            mock_process.terminate.assert_called_once()


class TestWatchManuscriptParameters:
    """Tests for watch_manuscript parameter defaults."""

    def test_interval_default(self, tmp_path):
        """Verify interval parameter has default value."""
        compile_script = tmp_path / "compile"
        compile_script.write_text("#!/bin/bash")

        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = ""

        with patch("subprocess.Popen", return_value=mock_process):
            # Should work without interval parameter
            watch_manuscript(tmp_path)

    def test_timeout_none_by_default(self, tmp_path):
        """Verify timeout defaults to None."""
        compile_script = tmp_path / "compile"
        compile_script.write_text("#!/bin/bash")

        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = ""

        with patch("subprocess.Popen", return_value=mock_process):
            watch_manuscript(tmp_path)

            mock_process.wait.assert_called_once_with(timeout=None)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
