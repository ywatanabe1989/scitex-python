#!/usr/bin/env python3
# Time-stamp: "2026-01-05"
# File: ./tests/scitex/session/test__lifecycle.py

"""Tests for session lifecycle functions (start, close, running2finished)."""

import argparse
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Required for scitex.session module
pytest.importorskip("natsort")
pytest.importorskip("h5py")
pytest.importorskip("zarr")


class TestFormatDiffTime:
    """Tests for _format_diff_time helper function."""

    def test_format_seconds_only(self):
        """Test formatting time difference with seconds only."""
        from scitex.session._lifecycle import _format_diff_time

        diff = timedelta(seconds=45)
        result = _format_diff_time(diff)
        assert result == "00:00:45"

    def test_format_minutes_and_seconds(self):
        """Test formatting time difference with minutes and seconds."""
        from scitex.session._lifecycle import _format_diff_time

        diff = timedelta(minutes=5, seconds=30)
        result = _format_diff_time(diff)
        assert result == "00:05:30"

    def test_format_hours_minutes_seconds(self):
        """Test formatting time difference with hours, minutes, seconds."""
        from scitex.session._lifecycle import _format_diff_time

        diff = timedelta(hours=2, minutes=15, seconds=45)
        result = _format_diff_time(diff)
        assert result == "02:15:45"

    def test_format_zero_time(self):
        """Test formatting zero time difference."""
        from scitex.session._lifecycle import _format_diff_time

        diff = timedelta(seconds=0)
        result = _format_diff_time(diff)
        assert result == "00:00:00"

    def test_format_large_hours(self):
        """Test formatting large hour values."""
        from scitex.session._lifecycle import _format_diff_time

        diff = timedelta(hours=100, minutes=30, seconds=15)
        result = _format_diff_time(diff)
        assert result == "100:30:15"


class TestSimplifyRelativePath:
    """Tests for _simplify_relative_path helper function."""

    def test_simplify_running_path(self):
        """Test simplifying path with RUNNING directory."""
        from scitex.session._lifecycle import _simplify_relative_path

        # Use a path relative to current working directory for consistency
        cwd = os.getcwd()
        sdir = os.path.join(
            cwd, "scripts/experiment/RUNNING/2024Y-09M-12D-02h44m40s_GlBZ"
        )
        result = _simplify_relative_path(sdir)

        # Should remove RUNNING/ and date-time pattern
        assert "RUNNING" not in result
        assert "2024Y-09M-12D" not in result

    def test_simplify_scripts_path(self):
        """Test simplifying path with scripts directory."""
        from scitex.session._lifecycle import _simplify_relative_path

        cwd = os.getcwd()
        sdir = os.path.join(cwd, "scripts/test/RUNNING/2024Y-01M-01D-00h00m00s_XXXX")
        result = _simplify_relative_path(sdir)

        # Should replace scripts/ with ./scripts/
        assert "./scripts/" in result or "scripts/" in result


class TestGetDebugMode:
    """Tests for _get_debug_mode helper function."""

    def test_debug_mode_file_not_exists(self):
        """Test debug mode when config file doesn't exist."""
        from scitex.session._lifecycle import _get_debug_mode

        with patch("os.path.exists", return_value=False):
            result = _get_debug_mode()
            assert result is False

    def test_debug_mode_returns_bool(self):
        """Test debug mode returns boolean."""
        from scitex.session._lifecycle import _get_debug_mode

        result = _get_debug_mode()
        assert isinstance(result, bool)


class TestGetScitexVersion:
    """Tests for _get_scitex_version helper function."""

    def test_get_version_returns_string(self):
        """Test version returns string."""
        from scitex.session._lifecycle import _get_scitex_version

        result = _get_scitex_version()
        assert isinstance(result, str)

    def test_get_version_not_empty(self):
        """Test version is not empty."""
        from scitex.session._lifecycle import _get_scitex_version

        result = _get_scitex_version()
        assert len(result) > 0


class TestInitializeEnv:
    """Tests for _initialize_env helper function."""

    def test_initialize_env_returns_id_and_pid(self):
        """Test initialization returns ID and PID."""
        from scitex.session._lifecycle import _initialize_env

        ID, PID = _initialize_env(IS_DEBUG=False)

        assert isinstance(ID, str)
        assert len(ID) > 0
        assert isinstance(PID, int)
        assert PID == os.getpid()

    def test_initialize_env_debug_mode(self):
        """Test initialization in debug mode."""
        from scitex.session._lifecycle import _initialize_env

        ID, PID = _initialize_env(IS_DEBUG=True)

        assert ID.startswith("DEBUG_")
        assert isinstance(PID, int)


class TestSetupConfigs:
    """Tests for _setup_configs helper function."""

    def test_setup_configs_basic(self):
        """Test basic configuration setup."""
        from scitex.session._lifecycle import _setup_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "RUNNING", "test_id")
            os.makedirs(sdir)

            configs = _setup_configs(
                IS_DEBUG=False,
                ID="test_id",
                PID=12345,
                file="/tmp/test.py",
                sdir=sdir,
                relative_sdir="./test_out/RUNNING/test_id",
                verbose=False,
            )

            assert configs["ID"] == "test_id"
            assert configs["PID"] == 12345
            assert "START_DATETIME" in configs
            assert isinstance(configs["START_DATETIME"], datetime)

    def test_setup_configs_sdir_paths(self):
        """Test SDIR_OUT and SDIR_RUN are set correctly."""
        from scitex.session._lifecycle import _setup_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "script_out", "RUNNING", "test_id")
            os.makedirs(sdir)

            configs = _setup_configs(
                IS_DEBUG=False,
                ID="test_id",
                PID=12345,
                file="/tmp/test.py",
                sdir=sdir,
                relative_sdir="./script_out/RUNNING/test_id",
                verbose=False,
            )

            assert configs["SDIR_RUN"] == Path(sdir)
            # SDIR_OUT should be the parent of RUNNING
            assert "RUNNING" not in str(configs["SDIR_OUT"])


class TestSetupMatplotlib:
    """Tests for _setup_matplotlib helper function."""

    def test_setup_matplotlib_with_none(self):
        """Test matplotlib setup with None plt."""
        from scitex.session._lifecycle import _setup_matplotlib

        plt_result, colors = _setup_matplotlib(plt=None)

        assert plt_result is None
        assert colors is None

    def test_setup_matplotlib_with_plt(self):
        """Test matplotlib setup with actual pyplot."""
        import matplotlib.pyplot as plt

        from scitex.session._lifecycle import _setup_matplotlib

        plt_result, colors = _setup_matplotlib(plt=plt)

        assert plt_result is not None
        assert colors is not None
        # COLORS is a DotDict, check dict-like behavior
        assert hasattr(colors, "__getitem__")
        assert hasattr(colors, "keys")
        # Check that gray alias is added
        if "grey" in colors:
            assert "gray" in colors


class TestArgsToStr:
    """Tests for _args_to_str helper function."""

    def test_args_to_str_empty(self):
        """Test args to string with empty dict."""
        from scitex.session._lifecycle import _args_to_str

        result = _args_to_str({})
        assert result == ""

    def test_args_to_str_with_values(self):
        """Test args to string with values."""
        from scitex.session._lifecycle import _args_to_str

        args = {"key1": "value1", "key2": 42}
        result = _args_to_str(args)

        assert "key1" in result
        assert "value1" in result
        assert "key2" in result
        assert "42" in result

    def test_args_to_str_formatting(self):
        """Test args to string has proper formatting."""
        from scitex.session._lifecycle import _args_to_str

        args = {"a": 1, "bb": 2}
        result = _args_to_str(args)

        # Should have newlines separating entries
        assert "\n" in result or len(args) == 1


class TestEscapeAnsiFromLogFiles:
    """Tests for _escape_ansi_from_log_files helper function."""

    def test_escape_ansi_removes_codes(self):
        """Test ANSI escape codes are removed."""
        from scitex.session._lifecycle import _escape_ansi_from_log_files

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")

            # Write file with ANSI codes
            with open(log_file, "w") as f:
                f.write("\x1b[31mRed text\x1b[0m and normal text")

            _escape_ansi_from_log_files([log_file])

            # Read back and verify codes are removed
            with open(log_file) as f:
                content = f.read()

            assert "\x1b[31m" not in content
            assert "\x1b[0m" not in content
            assert "Red text" in content
            assert "normal text" in content

    def test_escape_ansi_empty_file_list(self):
        """Test with empty file list."""
        from scitex.session._lifecycle import _escape_ansi_from_log_files

        # Should not raise
        _escape_ansi_from_log_files([])


class TestProcessTimestamp:
    """Tests for _process_timestamp helper function."""

    def test_process_timestamp_adds_end_time(self):
        """Test processing adds end datetime."""
        from scitex.session._lifecycle import _process_timestamp

        config = {"START_DATETIME": datetime.now() - timedelta(minutes=5)}

        result = _process_timestamp(config, verbose=False)

        assert "END_DATETIME" in result
        assert isinstance(result["END_DATETIME"], datetime)
        assert result["END_DATETIME"] > result["START_DATETIME"]

    def test_process_timestamp_calculates_duration(self):
        """Test processing calculates run duration."""
        from scitex.session._lifecycle import _process_timestamp

        start_time = datetime.now() - timedelta(hours=1, minutes=30, seconds=45)
        config = {"START_DATETIME": start_time}

        result = _process_timestamp(config, verbose=False)

        assert "RUN_DURATION" in result
        assert isinstance(result["RUN_DURATION"], str)
        # Should be in HH:MM:SS format
        parts = result["RUN_DURATION"].split(":")
        assert len(parts) == 3


class TestRunning2Finished:
    """Tests for running2finished function."""

    def test_running2finished_success_status(self):
        """Test moving to FINISHED_SUCCESS with exit_status=0."""
        from scitex.session._lifecycle import running2finished

        with tempfile.TemporaryDirectory() as tmpdir:
            running_dir = os.path.join(tmpdir, "script_out", "RUNNING", "test_id")
            os.makedirs(running_dir)

            # Create a test file
            test_file = os.path.join(running_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            config = {"SDIR_RUN": Path(running_dir)}

            result = running2finished(
                config, exit_status=0, remove_src_dir=True, max_wait=5
            )

            assert "FINISHED_SUCCESS" in str(result["SDIR_RUN"])

    def test_running2finished_error_status(self):
        """Test moving to FINISHED_ERROR with exit_status=1."""
        from scitex.session._lifecycle import running2finished

        with tempfile.TemporaryDirectory() as tmpdir:
            running_dir = os.path.join(tmpdir, "script_out", "RUNNING", "test_id")
            os.makedirs(running_dir)

            # Create a test file
            test_file = os.path.join(running_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            config = {"SDIR_RUN": Path(running_dir)}

            result = running2finished(
                config, exit_status=1, remove_src_dir=True, max_wait=5
            )

            assert "FINISHED_ERROR" in str(result["SDIR_RUN"])

    def test_running2finished_no_status(self):
        """Test moving to FINISHED with exit_status=None."""
        from scitex.session._lifecycle import running2finished

        with tempfile.TemporaryDirectory() as tmpdir:
            running_dir = os.path.join(tmpdir, "script_out", "RUNNING", "test_id")
            os.makedirs(running_dir)

            # Create a test file
            test_file = os.path.join(running_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            config = {"SDIR_RUN": Path(running_dir)}

            result = running2finished(
                config, exit_status=None, remove_src_dir=True, max_wait=5
            )

            assert "FINISHED/" in str(result["SDIR_RUN"]) or "FINISHED\\" in str(
                result["SDIR_RUN"]
            )

    def test_running2finished_copies_files(self):
        """Test files are copied to destination."""
        from scitex.session._lifecycle import running2finished

        with tempfile.TemporaryDirectory() as tmpdir:
            running_dir = os.path.join(tmpdir, "script_out", "RUNNING", "test_id")
            os.makedirs(running_dir)

            # Create test files
            with open(os.path.join(running_dir, "file1.txt"), "w") as f:
                f.write("content1")

            sub_dir = os.path.join(running_dir, "subdir")
            os.makedirs(sub_dir)
            with open(os.path.join(sub_dir, "file2.txt"), "w") as f:
                f.write("content2")

            config = {"SDIR_RUN": Path(running_dir)}

            result = running2finished(
                config, exit_status=0, remove_src_dir=False, max_wait=5
            )

            # Check files exist in destination
            dest_dir = str(result["SDIR_RUN"])
            assert os.path.exists(os.path.join(dest_dir, "file1.txt"))
            assert os.path.exists(os.path.join(dest_dir, "subdir", "file2.txt"))

    def test_running2finished_removes_source(self):
        """Test source directory is removed when remove_src_dir=True."""
        from scitex.session._lifecycle import running2finished

        with tempfile.TemporaryDirectory() as tmpdir:
            running_dir = os.path.join(tmpdir, "script_out", "RUNNING", "test_id")
            os.makedirs(running_dir)

            with open(os.path.join(running_dir, "file.txt"), "w") as f:
                f.write("content")

            config = {"SDIR_RUN": Path(running_dir)}

            running2finished(config, exit_status=0, remove_src_dir=True, max_wait=5)

            # Source should be removed
            assert not os.path.exists(running_dir)


class TestStartFunction:
    """Tests for start() function."""

    def test_start_returns_tuple(self):
        """Test start returns correct tuple structure."""
        from scitex.session._lifecycle import start

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "test_out", "RUNNING", "test_session")
            os.makedirs(sdir, exist_ok=True)

            result = start(
                sys=None,
                plt=None,
                file="/tmp/test.py",
                sdir=sdir,
                verbose=False,
            )

            assert len(result) == 6
            CONFIG, stdout, stderr, plt_result, COLORS, rng = result

            # CONFIG should be a DotDict
            assert hasattr(CONFIG, "ID")
            assert hasattr(CONFIG, "PID")

            # Without sys, stdout/stderr should be None
            assert stdout is None
            assert stderr is None

    def test_start_creates_sdir(self):
        """Test start creates save directory."""
        from scitex.session._lifecycle import start

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_script.py")
            with open(test_file, "w") as f:
                f.write("# test")

            result = start(
                sys=None,
                plt=None,
                file=test_file,
                sdir=None,
                verbose=False,
            )

            CONFIG = result[0]
            assert os.path.exists(str(CONFIG["SDIR_RUN"]))

            # Cleanup
            shutil.rmtree(str(CONFIG["SDIR_OUT"]), ignore_errors=True)

    def test_start_with_custom_sdir(self):
        """Test start with custom sdir."""
        from scitex.session._lifecycle import start

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_sdir = os.path.join(tmpdir, "custom_out", "RUNNING", "session_id")

            result = start(
                sys=None,
                plt=None,
                file="/tmp/test.py",
                sdir=custom_sdir,
                verbose=False,
            )

            CONFIG = result[0]
            assert str(CONFIG["SDIR_RUN"]) == custom_sdir

    def test_start_with_path_object_sdir(self):
        """Test start accepts Path object for sdir."""
        from scitex.session._lifecycle import start

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_sdir = Path(tmpdir) / "custom_out" / "RUNNING" / "session_id"

            result = start(
                sys=None,
                plt=None,
                file="/tmp/test.py",
                sdir=custom_sdir,
                verbose=False,
            )

            CONFIG = result[0]
            assert os.path.exists(str(CONFIG["SDIR_RUN"]))

    def test_start_with_sdir_suffix(self):
        """Test start with sdir_suffix."""
        from scitex.session._lifecycle import start

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_script.py")
            with open(test_file, "w") as f:
                f.write("# test")

            result = start(
                sys=None,
                plt=None,
                file=test_file,
                sdir=None,
                sdir_suffix="my_suffix",
                verbose=False,
            )

            CONFIG = result[0]
            assert "my_suffix" in str(CONFIG["SDIR_RUN"])

            # Cleanup
            shutil.rmtree(str(CONFIG["SDIR_OUT"]), ignore_errors=True)

    def test_start_with_args(self):
        """Test start with command line args."""
        from scitex.session._lifecycle import start

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "test_out", "RUNNING", "test_session")

            args = argparse.Namespace(param1="value1", param2=42)

            result = start(
                sys=None,
                plt=None,
                file="/tmp/test.py",
                sdir=sdir,
                args=args,
                verbose=False,
            )

            CONFIG = result[0]
            assert CONFIG["ARGS"]["param1"] == "value1"
            assert CONFIG["ARGS"]["param2"] == 42

    def test_start_registers_session(self):
        """Test start registers session with global manager."""
        from scitex.session._lifecycle import start
        from scitex.session._manager import get_global_session_manager

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "test_out", "RUNNING", "test_session")

            result = start(
                sys=None,
                plt=None,
                file="/tmp/test.py",
                sdir=sdir,
                verbose=False,
            )

            CONFIG = result[0]
            session_id = CONFIG["ID"]

            manager = get_global_session_manager()
            session_info = manager.get_session(session_id)

            assert session_info is not None
            assert session_info["status"] == "running"

    def test_start_initializes_rng(self):
        """Test start returns RandomStateManager."""
        from scitex.repro import RandomStateManager
        from scitex.session._lifecycle import start

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "test_out", "RUNNING", "test_session")

            result = start(
                sys=None,
                plt=None,
                file="/tmp/test.py",
                sdir=sdir,
                seed=123,
                verbose=False,
            )

            rng = result[5]
            assert isinstance(rng, RandomStateManager)

    def test_start_with_matplotlib(self):
        """Test start configures matplotlib."""
        import matplotlib.pyplot as plt

        from scitex.session._lifecycle import start

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "test_out", "RUNNING", "test_session")

            result = start(
                sys=None,
                plt=plt,
                file="/tmp/test.py",
                sdir=sdir,
                verbose=False,
            )

            plt_result = result[3]
            COLORS = result[4]

            assert plt_result is not None
            assert COLORS is not None
            # COLORS is a DotDict, check dict-like behavior
            assert hasattr(COLORS, "__getitem__")
            assert hasattr(COLORS, "keys")


class TestCloseFunction:
    """Tests for close() function."""

    def test_close_basic(self):
        """Test basic close functionality."""
        from scitex.session._lifecycle import close, start

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "test_out", "RUNNING", "test_session")

            CONFIG, _, _, _, _, _ = start(
                sys=None,
                plt=None,
                file="/tmp/test.py",
                sdir=sdir,
                verbose=False,
            )

            # Should not raise
            close(CONFIG, verbose=False, exit_status=0)

    def test_close_marks_session_closed(self):
        """Test close marks session as closed in manager."""
        from scitex.session._lifecycle import close, start
        from scitex.session._manager import get_global_session_manager

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "test_out", "RUNNING", "test_session")

            CONFIG, _, _, _, _, _ = start(
                sys=None,
                plt=None,
                file="/tmp/test.py",
                sdir=sdir,
                verbose=False,
            )

            session_id = CONFIG["ID"]
            close(CONFIG, verbose=False, exit_status=0)

            manager = get_global_session_manager()
            session_info = manager.get_session(session_id)

            assert session_info["status"] == "closed"

    def test_close_moves_to_finished(self):
        """Test close moves session to FINISHED directory."""
        from scitex.session._lifecycle import close, start

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "test_out", "RUNNING", "test_session")

            CONFIG, _, _, _, _, _ = start(
                sys=None,
                plt=None,
                file="/tmp/test.py",
                sdir=sdir,
                verbose=False,
            )

            close(CONFIG, verbose=False, exit_status=0)

            # Original RUNNING directory should be gone
            assert not os.path.exists(sdir)

    def test_close_with_different_exit_statuses(self):
        """Test close with different exit status values."""
        from scitex.session._lifecycle import close, start

        for exit_status in [0, 1, None]:
            with tempfile.TemporaryDirectory() as tmpdir:
                sdir = os.path.join(
                    tmpdir, "test_out", "RUNNING", f"test_session_{exit_status}"
                )

                CONFIG, _, _, _, _, _ = start(
                    sys=None,
                    plt=None,
                    file="/tmp/test.py",
                    sdir=sdir,
                    verbose=False,
                )

                # Should not raise
                close(CONFIG, verbose=False, exit_status=exit_status)


class TestClearPythonLogDir:
    """Tests for _clear_python_log_dir helper function."""

    def test_clear_existing_dir(self):
        """Test clearing existing log directory."""
        from scitex.session._lifecycle import _clear_python_log_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            os.makedirs(log_dir)

            # Create some files
            with open(os.path.join(log_dir, "test.log"), "w") as f:
                f.write("test")

            _clear_python_log_dir(log_dir)

            assert not os.path.exists(log_dir)

    def test_clear_nonexistent_dir(self):
        """Test clearing non-existent directory doesn't raise."""
        from scitex.session._lifecycle import _clear_python_log_dir

        # Should not raise
        _clear_python_log_dir("/nonexistent/path/that/does/not/exist")


class TestIntegration:
    """Integration tests for session lifecycle."""

    def test_full_session_lifecycle(self):
        """Test complete session lifecycle: start -> use -> close."""
        from scitex.session._lifecycle import close, start
        from scitex.session._manager import get_global_session_manager

        with tempfile.TemporaryDirectory() as tmpdir:
            sdir = os.path.join(tmpdir, "test_out", "RUNNING", "full_test")

            # Start session
            CONFIG, stdout, stderr, plt_result, COLORS, rng = start(
                sys=None,
                plt=None,
                file="/tmp/test.py",
                sdir=sdir,
                verbose=False,
            )

            session_id = CONFIG["ID"]

            # Verify session is running
            manager = get_global_session_manager()
            assert manager.get_session(session_id)["status"] == "running"

            # Close session
            close(CONFIG, verbose=False, exit_status=0)

            # Verify session is closed
            assert manager.get_session(session_id)["status"] == "closed"

    def test_multiple_sequential_sessions(self):
        """Test multiple sessions can be run sequentially."""
        from scitex.session._lifecycle import close, start

        session_ids = []

        for i in range(3):
            with tempfile.TemporaryDirectory() as tmpdir:
                sdir = os.path.join(tmpdir, "test_out", "RUNNING", f"session_{i}")

                CONFIG, _, _, _, _, _ = start(
                    sys=None,
                    plt=None,
                    file="/tmp/test.py",
                    sdir=sdir,
                    verbose=False,
                )

                session_ids.append(CONFIG["ID"])
                close(CONFIG, verbose=False, exit_status=0)

        # All session IDs should be unique
        assert len(set(session_ids)) == 3


class TestPrintHeader:
    """Tests for _print_header helper function."""

    def test_print_header_with_args(self):
        """Test print header with argparse namespace."""
        from scitex.dict import DotDict
        from scitex.session._lifecycle import _print_header

        args = argparse.Namespace(param1="value1", param2=42)
        configs = DotDict({"test": "value"})

        # Should not raise
        with patch("scitex.session._lifecycle._printc"):
            with patch("scitex.session._lifecycle.sleep"):
                _print_header(
                    ID="test_id",
                    PID=12345,
                    file="/tmp/test.py",
                    args=args,
                    configs=configs,
                    verbose=False,
                )

    def test_print_header_without_args(self):
        """Test print header without args."""
        from scitex.dict import DotDict
        from scitex.session._lifecycle import _print_header

        configs = DotDict({"test": "value"})

        # Should not raise
        with patch("scitex.session._lifecycle._printc"):
            with patch("scitex.session._lifecycle.sleep"):
                _print_header(
                    ID="test_id",
                    PID=12345,
                    file="/tmp/test.py",
                    args=None,
                    configs=configs,
                    verbose=False,
                )


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/session/_lifecycle.py
# --------------------------------------------------------------------------------
# (Source code reference maintained for sync_tests_with_source.sh)
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/session/_lifecycle.py
# --------------------------------------------------------------------------------
