#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/tests/scitex/verify/test__chain.py

"""Tests for scitex.clew._chain module."""

from scitex.clew import (
    ChainVerification,
    FileVerification,
    RunVerification,
    VerificationLevel,
    VerificationStatus,
    verify_file,
)


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.MISMATCH.value == "mismatch"
        assert VerificationStatus.MISSING.value == "missing"
        assert VerificationStatus.UNKNOWN.value == "unknown"


class TestVerificationLevel:
    """Tests for VerificationLevel enum."""

    def test_level_values(self):
        """Test that all expected level values exist."""
        assert VerificationLevel.CACHE.value == "cache"
        assert VerificationLevel.RERUN.value == "rerun"


class TestFileVerification:
    """Tests for FileVerification dataclass."""

    def test_file_verification_creation(self):
        """Test creating a FileVerification."""
        fv = FileVerification(
            path="/path/to/file.csv",
            role="input",
            expected_hash="abc123",
            current_hash="abc123",
            status=VerificationStatus.VERIFIED,
        )
        assert fv.path == "/path/to/file.csv"
        assert fv.role == "input"
        assert fv.is_verified is True

    def test_file_verification_mismatch(self):
        """Test FileVerification with mismatch."""
        fv = FileVerification(
            path="/path/to/file.csv",
            role="output",
            expected_hash="abc123",
            current_hash="def456",
            status=VerificationStatus.MISMATCH,
        )
        assert fv.is_verified is False

    def test_file_verification_missing(self):
        """Test FileVerification with missing file."""
        fv = FileVerification(
            path="/path/to/missing.csv",
            role="input",
            expected_hash="abc123",
            current_hash=None,
            status=VerificationStatus.MISSING,
        )
        assert fv.is_verified is False
        assert fv.current_hash is None


class TestRunVerification:
    """Tests for RunVerification dataclass."""

    def test_run_verification_creation(self):
        """Test creating a RunVerification."""
        rv = RunVerification(
            session_id="test_session",
            script_path="/path/to/script.py",
            status=VerificationStatus.VERIFIED,
            files=[],
            combined_hash_expected="hash1",
            combined_hash_current="hash1",
            level=VerificationLevel.CACHE,
        )
        assert rv.session_id == "test_session"
        assert rv.is_verified is True

    def test_run_verification_with_files(self):
        """Test RunVerification with file verifications."""
        files = [
            FileVerification(
                "/input.csv", "input", "h1", "h1", VerificationStatus.VERIFIED
            ),
            FileVerification(
                "/output.csv", "output", "h2", "h2", VerificationStatus.VERIFIED
            ),
        ]
        rv = RunVerification(
            session_id="test",
            script_path="/script.py",
            status=VerificationStatus.VERIFIED,
            files=files,
            combined_hash_expected=None,
            combined_hash_current=None,
        )
        assert len(rv.files) == 2
        assert len(rv.inputs) == 1
        assert len(rv.outputs) == 1

    def test_run_verification_mismatched_files(self):
        """Test RunVerification.mismatched_files property."""
        files = [
            FileVerification(
                "/good.csv", "input", "h1", "h1", VerificationStatus.VERIFIED
            ),
            FileVerification(
                "/bad.csv", "output", "h2", "h3", VerificationStatus.MISMATCH
            ),
        ]
        rv = RunVerification(
            session_id="test",
            script_path="/script.py",
            status=VerificationStatus.MISMATCH,
            files=files,
            combined_hash_expected=None,
            combined_hash_current=None,
        )
        assert len(rv.mismatched_files) == 1
        assert rv.mismatched_files[0].path == "/bad.csv"

    def test_run_verification_missing_files(self):
        """Test RunVerification.missing_files property."""
        files = [
            FileVerification(
                "/exists.csv", "input", "h1", "h1", VerificationStatus.VERIFIED
            ),
            FileVerification(
                "/missing.csv", "input", "h2", None, VerificationStatus.MISSING
            ),
        ]
        rv = RunVerification(
            session_id="test",
            script_path="/script.py",
            status=VerificationStatus.MISSING,
            files=files,
            combined_hash_expected=None,
            combined_hash_current=None,
        )
        assert len(rv.missing_files) == 1
        assert rv.missing_files[0].path == "/missing.csv"

    def test_run_verification_rerun_level(self):
        """Test RunVerification with rerun level."""
        rv = RunVerification(
            session_id="test",
            script_path="/script.py",
            status=VerificationStatus.VERIFIED,
            files=[],
            combined_hash_expected=None,
            combined_hash_current=None,
            level=VerificationLevel.RERUN,
        )
        assert rv.is_verified_from_scratch is True


class TestChainVerification:
    """Tests for ChainVerification dataclass."""

    def test_chain_verification_creation(self):
        """Test creating a ChainVerification."""
        cv = ChainVerification(
            target_file="/path/to/output.csv",
            runs=[],
            status=VerificationStatus.VERIFIED,
        )
        assert cv.target_file == "/path/to/output.csv"
        assert cv.is_verified is True

    def test_chain_verification_with_runs(self):
        """Test ChainVerification with multiple runs."""
        runs = [
            RunVerification(
                "s1",
                "/p1.py",
                VerificationStatus.VERIFIED,
                [],
                None,
                None,
            ),
            RunVerification(
                "s2",
                "/p2.py",
                VerificationStatus.VERIFIED,
                [],
                None,
                None,
            ),
        ]
        cv = ChainVerification(
            target_file="/output.csv",
            runs=runs,
            status=VerificationStatus.VERIFIED,
        )
        assert len(cv.runs) == 2

    def test_chain_verification_failed_runs(self):
        """Test ChainVerification.failed_runs property."""
        runs = [
            RunVerification(
                "s1",
                "/p1.py",
                VerificationStatus.VERIFIED,
                [],
                None,
                None,
            ),
            RunVerification(
                "s2",
                "/p2.py",
                VerificationStatus.MISMATCH,
                [],
                None,
                None,
            ),
        ]
        cv = ChainVerification(
            target_file="/output.csv",
            runs=runs,
            status=VerificationStatus.MISMATCH,
        )
        assert len(cv.failed_runs) == 1
        assert cv.failed_runs[0].session_id == "s2"


class TestVerifyFile:
    """Tests for verify_file function."""

    def test_verify_file_match(self, tmp_path):
        """Test verifying a file that matches expected hash."""
        from scitex.clew import hash_file

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        expected_hash = hash_file(test_file)
        result = verify_file(test_file, expected_hash)

        assert result.status == VerificationStatus.VERIFIED
        assert result.is_verified is True

    def test_verify_file_mismatch(self, tmp_path):
        """Test verifying a file that doesn't match expected hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = verify_file(test_file, "wronghash1234567890123456789012")

        assert result.status == VerificationStatus.MISMATCH
        assert result.is_verified is False

    def test_verify_file_missing(self, tmp_path):
        """Test verifying a missing file."""
        result = verify_file(tmp_path / "missing.txt", "somehash")

        assert result.status == VerificationStatus.MISSING
        assert result.is_verified is False
        assert result.current_hash is None


# EOF
