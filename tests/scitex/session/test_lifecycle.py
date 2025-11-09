#!/usr/bin/env python3
"""
Tests for scitex.session lifecycle functions with pathlib.Path support
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the source directory to the path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

import scitex.session as session
from scitex.dict import DotDict


class TestSessionLifecycle:
    """Test session start and close functionality with Path support."""

    def test_session_start_with_string_sdir(self):
        """Test session.start() with string sdir parameter (backward compatibility)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sdir_str = os.path.join(tmpdir, "test_session")
            
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,  # Don't redirect output in tests
                plt=None,  # Don't setup matplotlib
                sdir=sdir_str,
                verbose=False
            )
            
            assert isinstance(CONFIG, DotDict)
            assert "SDIR" in CONFIG
            assert "SDIR_PATH" in CONFIG
            assert isinstance(CONFIG["SDIR"], str)
            assert isinstance(CONFIG["SDIR_PATH"], Path)
            assert CONFIG["SDIR"] == str(CONFIG["SDIR_PATH"])
            assert CONFIG["SDIR"].startswith(tmpdir)

    def test_session_start_with_path_sdir(self):
        """Test session.start() with Path sdir parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sdir_path = Path(tmpdir) / "test_session_path"
            
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,
                plt=None,
                sdir=sdir_path,
                verbose=False
            )
            
            assert isinstance(CONFIG, DotDict)
            assert "SDIR" in CONFIG
            assert "SDIR_PATH" in CONFIG
            assert isinstance(CONFIG["SDIR"], str)
            assert isinstance(CONFIG["SDIR_PATH"], Path)
            assert CONFIG["SDIR"] == str(sdir_path)
            assert CONFIG["SDIR_PATH"] == sdir_path

    def test_session_start_auto_sdir(self):
        """Test session.start() with automatic sdir generation."""
        CONFIG, _, _, _, _, _ = session.start(
            sys=None,
            plt=None,
            sdir=None,  # Auto-generate
            verbose=False
        )
        
        assert isinstance(CONFIG, DotDict)
        assert "SDIR" in CONFIG
        assert "SDIR_PATH" in CONFIG
        assert isinstance(CONFIG["SDIR"], str)
        assert isinstance(CONFIG["SDIR_PATH"], Path)
        assert CONFIG["SDIR"] == str(CONFIG["SDIR_PATH"])
        # Auto-generated path should contain "RUNNING"
        assert "RUNNING" in CONFIG["SDIR"]

    def test_session_config_path_fields(self):
        """Test that configuration contains all expected path fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sdir_path = Path(tmpdir) / "config_test"
            
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,
                plt=None,
                sdir=sdir_path,
                verbose=False
            )
            
            # Check all path-related fields exist
            path_fields = ["SDIR", "SDIR_PATH", "REL_SDIR", "REL_SDIR_PATH", "FILE", "FILE_PATH"]
            for field in path_fields:
                assert field in CONFIG, f"Missing config field: {field}"
            
            # Check types
            assert isinstance(CONFIG["SDIR"], str)
            assert isinstance(CONFIG["SDIR_PATH"], Path)
            assert isinstance(CONFIG["REL_SDIR"], str)
            assert isinstance(CONFIG["REL_SDIR_PATH"], Path)
            
            # FILE_PATH might be None in some cases, so check conditionally
            if CONFIG["FILE_PATH"] is not None:
                assert isinstance(CONFIG["FILE_PATH"], Path)

    def test_session_start_with_sdir_suffix(self):
        """Test session.start() with sdir_suffix parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sdir_path = Path(tmpdir) / "base_session"
            suffix = "test_suffix"
            
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,
                plt=None,
                sdir=sdir_path,
                sdir_suffix=suffix,
                verbose=False
            )
            
            # Should contain the suffix
            assert suffix in CONFIG["SDIR"]
            assert suffix in str(CONFIG["SDIR_PATH"])

    def test_path_consistency(self):
        """Test that Path and string versions are always consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sdir_path = Path(tmpdir) / "consistency_test"
            
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,
                plt=None,
                sdir=sdir_path,
                verbose=False
            )
            
            # Test SDIR consistency
            assert CONFIG["SDIR"] == str(CONFIG["SDIR_PATH"])
            
            # Test REL_SDIR consistency
            assert CONFIG["REL_SDIR"] == str(CONFIG["REL_SDIR_PATH"])
            
            # Test FILE consistency (if FILE_PATH exists)
            if CONFIG["FILE_PATH"] is not None and CONFIG["FILE"] is not None:
                assert CONFIG["FILE"] == str(CONFIG["FILE_PATH"])

    def test_mixed_path_usage_patterns(self):
        """Test realistic mixed usage of Path and string configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            session_dir = base_dir / "mixed_usage"
            
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,
                plt=None,
                sdir=session_dir,
                verbose=False
            )
            
            # Test that we can use Path objects for subdirectories
            subdir = CONFIG["SDIR_PATH"] / "subdir"
            assert isinstance(subdir, Path)
            
            # Test that we can mix with string operations
            log_file = CONFIG["SDIR"] + "/test.log"
            assert isinstance(log_file, str)
            
            # Test that paths work with os.path operations
            joined = os.path.join(CONFIG["SDIR"], "test.txt")
            assert isinstance(joined, str)
            assert CONFIG["SDIR"] in joined


class TestSessionClose:
    """Test session.close() functionality."""

    def test_session_close_basic(self):
        """Test basic session.close() functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sdir_path = Path(tmpdir) / "close_test"
            
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,
                plt=None,
                sdir=sdir_path,
                verbose=False
            )
            
            # Should not raise an error
            session.close(CONFIG, verbose=False)
            
            # Check that FINISHED directory was created
            finished_dir = str(sdir_path).replace("RUNNING/", "FINISHED/")
            assert os.path.exists(finished_dir), f"FINISHED directory not created: {finished_dir}"

    def test_session_close_with_exit_status(self):
        """Test session.close() with different exit statuses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sdir_path = Path(tmpdir) / "exit_status_test"
            
            # Test success status
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,
                plt=None,
                sdir=sdir_path,
                verbose=False
            )
            
            session.close(CONFIG, exit_status=0, verbose=False)
            success_dir = str(sdir_path).replace("RUNNING/", "FINISHED_SUCCESS/")
            assert os.path.exists(success_dir)

    def test_session_running2finished(self):
        """Test the running2finished function directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sdir_path = Path(tmpdir) / "running2finished_test"
            
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,
                plt=None,
                sdir=sdir_path,
                verbose=False
            )
            
            # Test running2finished function
            updated_config = session.running2finished(CONFIG, exit_status=None)
            
            # Should update the SDIR to FINISHED
            assert "FINISHED" in updated_config["SDIR"]
            assert "RUNNING" not in updated_config["SDIR"]


class TestSessionIntegration:
    """Integration tests for session functionality."""

    def test_session_full_lifecycle_with_paths(self):
        """Test complete session lifecycle with Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "full_lifecycle"
            
            # Start session
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,
                plt=None,
                sdir=session_dir,
                verbose=False
            )
            
            # Verify configuration setup
            assert CONFIG["SDIR_PATH"].exists()
            assert isinstance(CONFIG["SDIR_PATH"], Path)
            
            # Create some test files using Path objects
            test_file = CONFIG["SDIR_PATH"] / "test_data.txt"
            test_file.write_text("test content")
            assert test_file.exists()
            
            # Close session
            session.close(CONFIG, verbose=False)
            
            # Verify files were moved to FINISHED directory
            finished_dir = Path(str(CONFIG["SDIR"]).replace("RUNNING/", "FINISHED/"))
            finished_test_file = finished_dir / "test_data.txt"
            assert finished_test_file.exists()
            assert finished_test_file.read_text() == "test content"

    def test_backward_compatibility(self):
        """Test that existing string-based code still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use string path (old style)
            sdir_str = os.path.join(tmpdir, "backward_compat")
            
            CONFIG, _, _, _, _, _ = session.start(
                sys=None,
                plt=None,
                sdir=sdir_str,
                verbose=False
            )
            
            # Old-style string operations should still work
            log_path = CONFIG["SDIR"] + "/test.log"
            config_path = os.path.join(CONFIG["SDIR"], "config.yaml")
            
            # But new Path objects should also be available
            path_log = CONFIG["SDIR_PATH"] / "test.log"
            path_config = CONFIG["SDIR_PATH"] / "config.yaml"
            
            # Both should point to the same location
            assert log_path == str(path_log)
            assert config_path == str(path_config)


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__)])