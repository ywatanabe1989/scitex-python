#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/tests/scitex/verify/test__db.py

"""Tests for scitex.clew._db module."""

import pytest

from scitex.clew import VerificationDB


class TestVerificationDB:
    """Tests for VerificationDB class."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_verification.db"
        return VerificationDB(db_path)

    def test_init_creates_database(self, tmp_path):
        """Test that initialization creates the database file."""
        db_path = tmp_path / "test.db"
        db = VerificationDB(db_path)
        assert db_path.exists()
        assert db is not None

    def test_init_creates_tables(self, db):
        """Test that initialization creates required tables."""
        with db._connect() as conn:
            # Check runs table exists
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
            ).fetchone()
            assert result is not None

            # Check file_hashes table exists
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='file_hashes'"
            ).fetchone()
            assert result is not None


class TestRunOperations:
    """Tests for run-related database operations."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_verification.db"
        return VerificationDB(db_path)

    def test_add_run(self, db):
        """Test adding a run."""
        db.add_run(
            session_id="test_session_001",
            script_path="/path/to/script.py",
            script_hash="abc123def456",
        )

        run = db.get_run("test_session_001")
        assert run is not None
        assert run["session_id"] == "test_session_001"
        assert run["script_path"] == "/path/to/script.py"
        assert run["script_hash"] == "abc123def456"

    def test_add_run_with_metadata(self, db):
        """Test adding a run with metadata."""
        metadata = {"key": "value", "number": 42}
        db.add_run(
            session_id="test_session_002",
            script_path="/path/to/script.py",
            metadata=metadata,
        )

        run = db.get_run("test_session_002")
        assert run is not None

    def test_add_run_with_parent(self, db):
        """Test adding a run with parent session."""
        db.add_run(session_id="parent_001", script_path="/path/parent.py")
        db.add_run(
            session_id="child_001",
            script_path="/path/child.py",
            parent_session="parent_001",
        )

        run = db.get_run("child_001")
        assert run["parent_session"] == "parent_001"

    def test_get_run_not_found(self, db):
        """Test getting a non-existent run."""
        result = db.get_run("nonexistent")
        assert result is None

    def test_finish_run_status(self, db):
        """Test finishing run with status."""
        db.add_run(session_id="test_session", script_path="/path/script.py")
        db.finish_run("test_session", status="success", exit_code=0)

        run = db.get_run("test_session")
        assert run["status"] == "success"
        assert run["exit_code"] == 0

    def test_list_runs(self, db):
        """Test listing runs."""
        db.add_run(session_id="session_a", script_path="/path/a.py")
        db.add_run(session_id="session_b", script_path="/path/b.py")
        db.add_run(session_id="session_c", script_path="/path/c.py")

        runs = db.list_runs(limit=10)
        assert len(runs) == 3

    def test_list_runs_with_limit(self, db):
        """Test listing runs with limit."""
        for i in range(5):
            db.add_run(session_id=f"session_{i}", script_path=f"/path/{i}.py")

        runs = db.list_runs(limit=3)
        assert len(runs) == 3

    def test_list_runs_with_status_filter(self, db):
        """Test listing runs filtered by status."""
        db.add_run(session_id="success_1", script_path="/path/a.py")
        db.finish_run("success_1", status="success")
        db.add_run(session_id="failed_1", script_path="/path/b.py")
        db.finish_run("failed_1", status="failed")

        success_runs = db.list_runs(status="success")
        assert all(r["status"] == "success" for r in success_runs)


class TestFileHashOperations:
    """Tests for file hash-related database operations."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_verification.db"
        return VerificationDB(db_path)

    def test_add_file_hash(self, db):
        """Test adding a file hash."""
        db.add_run(session_id="test_session", script_path="/path/script.py")
        db.add_file_hash(
            session_id="test_session",
            file_path="/path/to/data.csv",
            hash_value="hash123",
            role="input",
        )

        hashes = db.get_file_hashes("test_session")
        assert "/path/to/data.csv" in hashes
        assert hashes["/path/to/data.csv"] == "hash123"

    def test_add_file_hash_multiple(self, db):
        """Test adding multiple file hashes."""
        db.add_run(session_id="test_session", script_path="/path/script.py")
        db.add_file_hash("test_session", "/path/input.csv", "hash1", "input")
        db.add_file_hash("test_session", "/path/output.csv", "hash2", "output")

        all_hashes = db.get_file_hashes("test_session")
        assert len(all_hashes) == 2

    def test_get_file_hashes_by_role(self, db):
        """Test getting file hashes filtered by role."""
        db.add_run(session_id="test_session", script_path="/path/script.py")
        db.add_file_hash("test_session", "/path/input.csv", "hash1", "input")
        db.add_file_hash("test_session", "/path/output.csv", "hash2", "output")

        inputs = db.get_file_hashes("test_session", role="input")
        assert len(inputs) == 1
        assert "/path/input.csv" in inputs

        outputs = db.get_file_hashes("test_session", role="output")
        assert len(outputs) == 1
        assert "/path/output.csv" in outputs

    def test_find_session_by_file(self, db):
        """Test finding sessions by file path."""
        db.add_run(session_id="session_1", script_path="/path/script.py")
        db.add_file_hash("session_1", "/shared/data.csv", "hash1", "output")

        db.add_run(session_id="session_2", script_path="/path/script.py")
        db.add_file_hash("session_2", "/shared/data.csv", "hash2", "input")

        sessions = db.find_session_by_file("/shared/data.csv", role="output")
        assert "session_1" in sessions


class TestChainOperations:
    """Tests for chain-related database operations."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_verification.db"
        return VerificationDB(db_path)

    def test_get_chain_single(self, db):
        """Test getting chain for a single run."""
        db.add_run(session_id="single", script_path="/path/script.py")

        chain = db.get_chain("single")
        assert chain == ["single"]

    def test_get_chain_with_parent(self, db):
        """Test getting chain with parent-child relationship."""
        db.add_run(session_id="parent", script_path="/path/parent.py")
        db.add_run(
            session_id="child",
            script_path="/path/child.py",
            parent_session="parent",
        )

        chain = db.get_chain("child")
        assert "child" in chain
        assert "parent" in chain

    def test_get_chain_multi_level(self, db):
        """Test getting chain with multiple levels."""
        db.add_run(session_id="grandparent", script_path="/path/gp.py")
        db.add_run(
            session_id="parent",
            script_path="/path/p.py",
            parent_session="grandparent",
        )
        db.add_run(
            session_id="child",
            script_path="/path/c.py",
            parent_session="parent",
        )

        chain = db.get_chain("child")
        assert len(chain) >= 3


class TestVerificationRecords:
    """Tests for verification record operations."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_verification.db"
        return VerificationDB(db_path)

    def test_record_verification(self, db):
        """Test recording a verification result."""
        db.add_run(session_id="test_session", script_path="/path/script.py")
        db.record_verification(
            session_id="test_session",
            level="cache",
            status="verified",
        )

        # Should not raise
        verification = db.get_latest_verification("test_session")
        assert verification is not None
        assert verification["level"] == "cache"
        assert verification["status"] == "verified"

    def test_record_verification_multiple(self, db):
        """Test recording multiple verification results."""
        import time

        db.add_run(session_id="test_session", script_path="/path/script.py")
        db.record_verification("test_session", "cache", "verified")
        time.sleep(0.01)  # Ensure different timestamp
        db.record_verification("test_session", "rerun", "verified")

        verification = db.get_latest_verification("test_session")
        # Latest verification should exist
        assert verification is not None


class TestDatabaseStats:
    """Tests for database statistics."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_verification.db"
        return VerificationDB(db_path)

    def test_stats(self, db):
        """Test getting database statistics."""
        db.add_run(session_id="s1", script_path="/p1.py")
        db.finish_run("s1", status="success")
        db.add_run(session_id="s2", script_path="/p2.py")
        db.finish_run("s2", status="failed")

        stats = db.stats()
        assert "total_runs" in stats
        assert stats["total_runs"] == 2

    def test_stats_empty_db(self, db):
        """Test stats on empty database."""
        stats = db.stats()
        assert stats["total_runs"] == 0


# EOF
