#!/usr/bin/env python3
# Timestamp: "2026-01-04 22:40:00 (Claude)"
# File: /tests/scitex/db/test__inspect.py

import os
import shutil
import sqlite3
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from scitex.db import inspect
from scitex.db._inspect import OptimizedInspector


class TestOptimizedInspector:
    """Test cases for OptimizedInspector class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test databases."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        # Cleanup
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    @pytest.fixture
    def db_path(self, temp_dir):
        """Get a temporary database path."""
        return os.path.join(temp_dir, "test.db")

    @pytest.fixture
    def sample_db(self, db_path):
        """Create a sample database with multiple tables."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create users table
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                age INTEGER
            )
        """)

        # Insert sample data
        users_data = [
            (1, "Alice", "alice@example.com", 30),
            (2, "Bob", "bob@example.com", 25),
            (3, "Charlie", "charlie@example.com", 35),
            (4, "David", "david@example.com", 28),
            (5, "Eve", "eve@example.com", 32),
        ]
        cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", users_data)

        # Create orders table
        cursor.execute("""
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                product TEXT,
                price REAL,
                data BLOB,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Insert order data
        orders_data = [
            (1, 1, "Laptop", 999.99, b"binary_data_1"),
            (2, 1, "Mouse", 29.99, b"binary_data_2"),
            (3, 2, "Keyboard", 79.99, b"binary_data_3"),
            (4, 3, "Monitor", 299.99, b"binary_data_4"),
        ]
        cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders_data)

        conn.commit()
        conn.close()

        return db_path

    def test_inspector_init(self, sample_db):
        """Test OptimizedInspector initialization."""
        inspector = OptimizedInspector(sample_db)
        assert inspector.db_path == sample_db
        inspector.close()

    def test_inspector_init_nonexistent(self, temp_dir):
        """Test OptimizedInspector initialization with non-existent database."""
        with pytest.raises(FileNotFoundError):
            OptimizedInspector(os.path.join(temp_dir, "nonexistent.db"))

    def test_context_manager(self, sample_db):
        """Test OptimizedInspector as context manager."""
        with OptimizedInspector(sample_db) as inspector:
            tables = inspector.get_table_names()
            assert len(tables) >= 2

    def test_get_table_names(self, sample_db):
        """Test retrieving table names."""
        with OptimizedInspector(sample_db) as inspector:
            tables = inspector.get_table_names()
            assert "users" in tables
            assert "orders" in tables
            assert len(tables) >= 2

    def test_get_table_info_batch(self, sample_db):
        """Test retrieving table structure information in batch."""
        with OptimizedInspector(sample_db) as inspector:
            info = inspector.get_table_info_batch(["users"])

            assert "users" in info
            assert len(info["users"]) == 4  # 4 columns in users table

            # Check column names
            column_names = [col["name"] for col in info["users"]]
            assert "id" in column_names
            assert "name" in column_names
            assert "email" in column_names
            assert "age" in column_names

            # Check primary key is detected
            id_col = [col for col in info["users"] if col["name"] == "id"][0]
            assert id_col["pk"] is True

    def test_get_table_stats_batch(self, sample_db):
        """Test retrieving table statistics in batch."""
        with OptimizedInspector(sample_db) as inspector:
            stats = inspector.get_table_stats_batch(["users"], sample_size=3)

            assert "users" in stats
            assert stats["users"]["row_count"] == 5
            assert len(stats["users"]["sample_data"]) == 3
            assert "columns" in stats["users"]

    def test_get_table_stats_batch_skip_count(self, sample_db):
        """Test table stats with skip_count option."""
        with OptimizedInspector(sample_db) as inspector:
            stats = inspector.get_table_stats_batch(["users"], skip_count=True)

            assert stats["users"]["row_count"] == "Not counted"
            assert stats["users"]["is_approximate"] is None

    def test_inspect_fast_all_tables(self, sample_db):
        """Test fast inspection of all tables."""
        with OptimizedInspector(sample_db) as inspector:
            results = inspector.inspect_fast(verbose=False)

            assert len(results) >= 2
            table_names = [r["table_name"] for r in results]
            assert "users" in table_names
            assert "orders" in table_names

    def test_inspect_fast_specific_tables(self, sample_db):
        """Test fast inspection of specific tables."""
        with OptimizedInspector(sample_db) as inspector:
            results = inspector.inspect_fast(table_names=["users"], verbose=False)

            assert len(results) == 1
            assert results[0]["table_name"] == "users"
            assert results[0]["row_count"] == 5

    def test_inspect_fast_blob_handling(self, sample_db):
        """Test that BLOB data is handled correctly in fast inspection."""
        with OptimizedInspector(sample_db) as inspector:
            results = inspector.inspect_fast(
                table_names=["orders"], verbose=False, skip_blob_content=True
            )

            assert len(results) == 1
            # Check sample data exists
            if results[0]["sample_data"]:
                first_row = results[0]["sample_data"][0]
                # BLOB should be shown as placeholder
                assert "<BLOB" in first_row.get("data", "")

    def test_inspect_fast_with_blob_content(self, sample_db):
        """Test inspection with BLOB content included."""
        with OptimizedInspector(sample_db) as inspector:
            results = inspector.inspect_fast(
                table_names=["orders"], verbose=False, skip_blob_content=False
            )

            assert len(results) == 1
            if results[0]["sample_data"]:
                first_row = results[0]["sample_data"][0]
                # BLOB should be actual bytes
                assert isinstance(first_row.get("data"), bytes)


class TestInspectFunction:
    """Test cases for the inspect function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test databases."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    @pytest.fixture
    def db_path(self, temp_dir):
        """Get a temporary database path."""
        return os.path.join(temp_dir, "test.db")

    @pytest.fixture
    def sample_db(self, db_path):
        """Create a sample database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            )
        """)

        cursor.executemany(
            "INSERT INTO users VALUES (?, ?, ?)",
            [(1, "Alice", "alice@example.com"), (2, "Bob", "bob@example.com")],
        )

        conn.commit()
        conn.close()
        return db_path

    def test_inspect_function_basic(self, sample_db):
        """Test the main inspect function."""
        results = inspect(sample_db, verbose=False)

        assert len(results) >= 1
        assert results[0]["table_name"] == "users"
        assert results[0]["row_count"] == 2

    def test_inspect_function_verbose(self, sample_db, capsys):
        """Test inspect function with verbose output."""
        results = inspect(sample_db, verbose=True)

        assert len(results) >= 1
        captured = capsys.readouterr()
        assert "users" in captured.out

    def test_inspect_function_specific_tables(self, sample_db):
        """Test inspecting specific tables only."""
        results = inspect(sample_db, table_names=["users"], verbose=False)

        assert len(results) == 1
        assert results[0]["table_name"] == "users"

    def test_inspect_function_skip_count(self, sample_db):
        """Test inspect with skip_count option."""
        results = inspect(sample_db, skip_count=True, verbose=False)

        assert len(results) >= 1
        assert results[0]["row_count"] == "Not counted"

    def test_empty_database(self, db_path):
        """Test inspecting an empty database."""
        conn = sqlite3.connect(db_path)
        conn.close()

        results = inspect(db_path, verbose=False)
        assert len(results) == 0

    def test_table_with_no_data(self, db_path):
        """Test inspecting a table with no data."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE empty_table (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()
        conn.close()

        results = inspect(db_path, verbose=False)

        assert len(results) == 1
        assert results[0]["table_name"] == "empty_table"
        assert results[0]["row_count"] == 0
        assert len(results[0]["sample_data"]) == 0

    def test_complex_schema(self, db_path):
        """Test inspection with complex table schema."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE complex_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                score REAL CHECK(score >= 0 AND score <= 100)
            )
        """)

        cursor.execute(
            "INSERT INTO complex_table (name, score) VALUES (?, ?)", ("Test", 85.5)
        )

        conn.commit()
        conn.close()

        # Filter for just the table we created (sqlite_sequence may also exist)
        results = inspect(db_path, table_names=["complex_table"], verbose=False)

        assert len(results) == 1
        assert results[0]["table_name"] == "complex_table"
        assert len(results[0]["columns"]) == 5

    def test_nonexistent_database(self, temp_dir):
        """Test inspection of non-existent database raises error."""
        with pytest.raises(FileNotFoundError):
            inspect(os.path.join(temp_dir, "nonexistent.db"))


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
