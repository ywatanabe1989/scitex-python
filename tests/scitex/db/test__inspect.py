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
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/db/_inspect.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-09-11 19:04:11 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/db/_inspect.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import sqlite3
# from contextlib import contextmanager
# from typing import Any, Dict, List, Optional
# 
# 
# class OptimizedInspector:
#     """Optimized database inspector with connection reuse and efficient queries."""
# 
#     def __init__(self, db_path: str):
#         if not os.path.exists(db_path):
#             raise FileNotFoundError(f"Database file not found: {db_path}")
#         self.db_path = db_path
#         self._conn = None
#         self._cursor = None
# 
#     @contextmanager
#     def connection(self):
#         """Context manager for database connection reuse."""
#         if self._conn is None:
#             self._conn = sqlite3.connect(self.db_path)
#             self._conn.row_factory = sqlite3.Row  # Enable column access by name
#             self._cursor = self._conn.cursor()
#         try:
#             yield self._cursor
#         except Exception:
#             self._conn.rollback()
#             raise
# 
#     def close(self):
#         """Close the database connection."""
#         if self._conn:
#             self._conn.close()
#             self._conn = None
#             self._cursor = None
# 
#     def __enter__(self):
#         return self
# 
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.close()
# 
#     def get_table_names(self) -> List[str]:
#         """Retrieves all table names from the database."""
#         with self.connection() as cursor:
#             cursor.execute(
#                 "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
#             )
#             return [row[0] for row in cursor.fetchall()]
# 
#     def get_table_info_batch(self, table_names: List[str]) -> Dict[str, List[Dict]]:
#         """Get table info for multiple tables in one go.
# 
#         Returns:
#             Dict mapping table_name to list of column info dictionaries
#         """
#         table_info = {}
# 
#         with self.connection() as cursor:
#             for table_name in table_names:
#                 # Get column info
#                 cursor.execute(f"PRAGMA table_info({table_name})")
#                 columns = cursor.fetchall()
# 
#                 # Get primary key info more efficiently
#                 cursor.execute(
#                     f"""
#                     SELECT name FROM pragma_table_info('{table_name}')
#                     WHERE pk > 0
#                 """
#                 )
#                 pk_columns = {row[0] for row in cursor.fetchall()}
# 
#                 # Build column info
#                 col_info = []
#                 for col in columns:
#                     col_dict = {
#                         "cid": col[0],
#                         "name": col[1],
#                         "type": col[2],
#                         "notnull": col[3],
#                         "default": col[4],
#                         "pk": col[1] in pk_columns,
#                     }
#                     col_info.append(col_dict)
# 
#                 table_info[table_name] = col_info
# 
#         return table_info
# 
#     def get_table_stats_batch(
#         self,
#         table_names: List[str],
#         sample_size: int = 5,
#         skip_count: bool = False,
#     ) -> Dict[str, Dict]:
#         """Get statistics for multiple tables efficiently.
# 
#         Args:
#             table_names: List of table names to inspect
#             sample_size: Number of sample rows to retrieve
#             skip_count: If True, skip the COUNT(*) query for performance
# 
#         Returns:
#             Dict mapping table_name to statistics dictionary
#         """
#         stats = {}
# 
#         with self.connection() as cursor:
#             for table_name in table_names:
#                 table_stats = {}
# 
#                 # Get sample data
#                 cursor.execute(f"SELECT * FROM {table_name} LIMIT {sample_size}")
#                 table_stats["columns"] = [desc[0] for desc in cursor.description]
#                 table_stats["sample_data"] = cursor.fetchall()
# 
#                 # Get row count (can be slow for large tables)
#                 if not skip_count:
#                     # Use approximate count if available
#                     cursor.execute(
#                         f"""
#                         SELECT COUNT(*) FROM (
#                             SELECT 1 FROM {table_name} LIMIT 100000
#                         )
#                     """
#                     )
#                     approx_count = cursor.fetchone()[0]
# 
#                     if approx_count < 100000:
#                         # Small table, get exact count
#                         cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
#                         table_stats["row_count"] = cursor.fetchone()[0]
#                         table_stats["is_approximate"] = False
#                     else:
#                         # Large table, use approximate count
#                         table_stats["row_count"] = f">{approx_count:,}"
#                         table_stats["is_approximate"] = True
#                 else:
#                     table_stats["row_count"] = "Not counted"
#                     table_stats["is_approximate"] = None
# 
#                 stats[table_name] = table_stats
# 
#         return stats
# 
#     def inspect_fast(
#         self,
#         table_names: Optional[List[str]] = None,
#         sample_size: int = 5,
#         skip_count: bool = False,
#         skip_blob_content: bool = True,
#         verbose: bool = True,
#     ) -> List[Dict[str, Any]]:
#         """Fast inspection of database tables.
# 
#         Args:
#             table_names: Tables to inspect (None for all)
#             sample_size: Number of sample rows
#             skip_count: Skip row counting for performance
#             skip_blob_content: Don't load BLOB content
#             verbose: Print results
# 
#         Returns:
#             List of inspection results
#         """
#         if table_names is None:
#             table_names = self.get_table_names()
# 
#         # Batch operations for efficiency
#         table_info = self.get_table_info_batch(table_names)
#         table_stats = self.get_table_stats_batch(table_names, sample_size, skip_count)
# 
#         results = []
#         for table_name in table_names:
#             result = {
#                 "table_name": table_name,
#                 "columns": table_info[table_name],
#                 "row_count": table_stats[table_name]["row_count"],
#                 "is_approximate": table_stats[table_name]["is_approximate"],
#                 "sample_data": [],
#             }
# 
#             # Format sample data
#             for row in table_stats[table_name]["sample_data"]:
#                 formatted_row = {}
#                 for i, col_name in enumerate(table_stats[table_name]["columns"]):
#                     value = row[i]
#                     if isinstance(value, bytes):
#                         if skip_blob_content:
#                             formatted_row[col_name] = f"<BLOB {len(value)} bytes>"
#                         else:
#                             formatted_row[col_name] = value
#                     else:
#                         formatted_row[col_name] = value
#                 result["sample_data"].append(formatted_row)
# 
#             results.append(result)
# 
#             if verbose:
#                 self._print_table_info(result)
# 
#         return results
# 
#     def _print_table_info(self, result: Dict):
#         """Pretty print table information."""
#         print(f"\n{'=' * 60}")
#         print(f"Table: {result['table_name']}")
#         print(
#             f"Rows: {result['row_count']}"
#             + (" (approximate)" if result["is_approximate"] else "")
#         )
#         print(f"Columns: {len(result['columns'])}")
#         print(f"-" * 60)
# 
#         # Check if we have sample data
#         if result["sample_data"]:
#             # Get all column names
#             all_cols = list(result["sample_data"][0].keys())
# 
#             # Separate data columns and metadata columns
#             data_cols = []
#             metadata_cols = []
#             for c in all_cols:
#                 if c.endswith(("_dtype", "_shape", "_compressed")):
#                     metadata_cols.append(c)
#                 else:
#                     data_cols.append(c)
# 
#             # Show first row in detail (like df.iloc[0])
#             print(f"\nFirst row (schema + data for all {len(all_cols)} columns):")
#             print(f"  {'Column':<40} | {'Type':<20} | Value")
#             print(f"  {'-' * 40}-|-{'-' * 20}-|-{'-' * 50}")
# 
#             first_row = result["sample_data"][0]
# 
#             # Show data columns first, then metadata columns
#             for key in data_cols + metadata_cols:
#                 value = first_row.get(key)
# 
#                 # Format value for display
#                 if isinstance(value, str) and len(value) > 50:
#                     display_value = value[:47] + "..."
#                 elif value is None:
#                     display_value = "NULL"
#                 else:
#                     display_value = str(value)
# 
#                 # Find column info
#                 col_type = ""
#                 constraints = []
#                 for col in result["columns"]:
#                     if col["name"] == key:
#                         col_type = col["type"]
#                         if col.get("pk"):
#                             constraints.append("PK")
#                         if col.get("notnull"):
#                             constraints.append("NOT NULL")
#                         break
# 
#                 # Format type with constraints
#                 if constraints:
#                     type_display = f"{col_type} ({', '.join(constraints)})"
#                 else:
#                     type_display = col_type
# 
#                 # Truncate key if too long
#                 display_key = key if len(key) <= 40 else key[:37] + "..."
# 
#                 # Print in column format
#                 print(f"  {display_key:<40} | {type_display:<20} | {display_value}")
# 
#             # If there are more rows, show a compact table view
#             if len(result["sample_data"]) > 1:
#                 max_row = min(3, len(result["sample_data"]))
#                 total_samples = len(result["sample_data"])
# 
#                 # Determine which columns to show - prioritize data cols but include metadata if few data cols
#                 if len(data_cols) >= 5:
#                     header_cols = data_cols[:5]
#                     col_type = "first 5 data columns"
#                 else:
#                     # Show all data cols plus some metadata cols
#                     header_cols = (
#                         data_cols + metadata_cols[: max(0, 5 - len(data_cols))]
#                     )
#                     col_type = f"{len(data_cols)} data + {len(header_cols) - len(data_cols)} metadata columns"
# 
#                 print(
#                     f"\nAdditional samples (rows 2-{max_row} of {total_samples}, {col_type}):"
#                 )
# 
#                 # Print header
#                 header = " | ".join(f"{col[:12]:<12}" for col in header_cols)
#                 print(f"  {header}")
#                 print(f"  {'-' * len(header)}")
# 
#                 # Print rows 2-3
#                 for row in result["sample_data"][1:3]:
#                     values = []
#                     for col in header_cols:
#                         val = str(row.get(col, ""))
#                         if len(val) > 12:
#                             val = val[:9] + "..."
#                         values.append(f"{val:<12}")
#                     print(f"  {' | '.join(values)}")
#         else:
#             # No data - show schema only
#             print("\nNo data in table. Schema:")
#             for col in result["columns"]:
#                 col_type = col["type"]
#                 constraints = []
#                 if col.get("pk"):
#                     constraints.append("PRIMARY KEY")
#                 if col.get("notnull"):
#                     constraints.append("NOT NULL")
#                 if col.get("default") is not None:
#                     constraints.append(f"DEFAULT {col['default']}")
# 
#                 constraint_str = f" ({', '.join(constraints)})" if constraints else ""
#                 print(f"  {col['name']}: {col_type}{constraint_str}")
# 
# 
# def inspect(
#     lpath_db: str,
#     table_names: Optional[List[str]] = None,
#     sample_size: int = 5,
#     skip_count: bool = False,
#     verbose: bool = True,
# ) -> List[Dict[str, Any]]:
#     """
#     Optimized database inspection.
# 
#     Example:
#     >>> inspect('path/to/database.db')
#     >>> inspect('path/to/database.db', ['table1'], skip_count=True)
# 
#     Args:
#         lpath_db: Path to the SQLite database file
#         table_names: List of table names to inspect (None for all)
#         sample_size: Number of sample rows to retrieve
#         skip_count: Skip row counting for large tables (much faster)
#         verbose: Print inspection results
# 
#     Returns:
#         List of inspection results
#     """
#     with OptimizedInspector(lpath_db) as inspector:
#         return inspector.inspect_fast(
#             table_names=table_names,
#             sample_size=sample_size,
#             skip_count=skip_count,
#             verbose=verbose,
#         )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/db/_inspect.py
# --------------------------------------------------------------------------------
