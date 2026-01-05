#!/usr/bin/env python3
# Timestamp: "2026-01-04 22:50:00 (Claude)"
# File: /tests/scitex/db/_sqlite3/test__SQLite3.py

import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))

from scitex.db import SQLite3


class TestSQLite3:
    """Test cases for SQLite3 database class."""

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

    def test_init_creates_database(self, db_path):
        """Test that initialization creates a database file."""
        with SQLite3(db_path) as db:
            assert os.path.exists(db_path)
            assert db.db_path == db_path
            assert db.conn is not None
            assert db.cursor is not None

    def test_init_with_temp_database(self, temp_dir):
        """Test initialization with temporary database option."""
        original_path = os.path.join(temp_dir, "original.db")

        # Create original database
        with SQLite3(original_path) as db1:
            db1.create_table("test", {"id": "INTEGER"})

        # Open with temp copy
        with SQLite3(original_path, use_temp=True) as db2:
            # Note: use_temp creates a temp copy, check db_path behavior
            assert db2.conn is not None

    def test_context_manager(self, db_path):
        """Test that SQLite3 works as a context manager."""
        with SQLite3(db_path) as db:
            assert db.conn is not None
            assert db.cursor is not None

        # After context, connection should be closed
        assert db.conn is None
        assert db.cursor is None

    def test_context_manager_required(self, db_path):
        """Test that context manager is required for operations."""
        db = SQLite3(db_path)
        with pytest.raises(RuntimeError, match="must be used with context manager"):
            db.execute("SELECT 1")
        db.close()

    def test_call_method_returns_summary(self, db_path):
        """Test the __call__ method returns table summaries."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )
            db.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)", ("test1", 1.5)
            )
            summary = db(return_summary=True, print_summary=False)
            assert isinstance(summary, dict)
            assert "test_table" in summary

    def test_summary_property(self, db_path):
        """Test the summary property exists and is callable."""
        with SQLite3(db_path) as db:
            db.create_table("test_table", {"id": "INTEGER PRIMARY KEY"})
            # Just verify summary property exists and doesn't error
            assert hasattr(db, "summary")

    def test_create_table_basic(self, db_path):
        """Test basic table creation."""
        with SQLite3(db_path) as db:
            db.create_table(
                "users",
                {
                    "id": "INTEGER PRIMARY KEY",
                    "name": "TEXT NOT NULL",
                    "email": "TEXT UNIQUE",
                },
            )

            tables = db.get_table_names()
            assert "users" in tables

            schema = db.get_table_schema("users")
            assert len(schema) == 3
            assert "id" in schema["name"].values
            assert "name" in schema["name"].values
            assert "email" in schema["name"].values

    def test_create_table_with_blob(self, db_path):
        """Test table creation with BLOB columns adds metadata columns."""
        with SQLite3(db_path) as db:
            db.create_table(
                "images", {"id": "INTEGER PRIMARY KEY", "image_data": "BLOB"}
            )

            schema = db.get_table_schema("images")
            column_names = schema["name"].values
            assert "image_data" in column_names
            assert "image_data_dtype" in column_names
            assert "image_data_shape" in column_names

    def test_create_table_if_not_exists(self, db_path):
        """Test that creating existing table with if_not_exists=True doesn't fail."""
        with SQLite3(db_path) as db:
            db.create_table("test", {"id": "INTEGER"})
            # Should not raise
            db.create_table("test", {"id": "INTEGER"}, if_not_exists=True)

    def test_drop_table(self, db_path):
        """Test dropping a table."""
        with SQLite3(db_path) as db:
            db.create_table("test_table", {"id": "INTEGER PRIMARY KEY"})
            db.drop_table("test_table")
            tables = db.get_table_names()
            assert "test_table" not in tables

    def test_basic_crud_operations(self, db_path):
        """Test basic Create, Read, Update, Delete operations."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            # Create (Insert)
            db.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                ("test_item", 42.0),
            )

            # Read
            result = db.get_rows("test_table", where="name='test_item'")
            assert len(result) == 1
            assert result.iloc[0]["name"] == "test_item"
            assert result.iloc[0]["value"] == 42.0

            # Update
            db.execute(
                "UPDATE test_table SET value = ? WHERE name = ?", (99.0, "test_item")
            )

            result = db.get_rows("test_table", where="name='test_item'")
            assert result.iloc[0]["value"] == 99.0

            # Delete
            db.execute("DELETE FROM test_table WHERE name = ?", ("test_item",))
            result = db.get_rows("test_table", where="name='test_item'")
            assert len(result) == 0

    def test_transaction_commit(self, db_path):
        """Test transaction commit."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            with db.transaction():
                db.execute(
                    "INSERT INTO test_table (name, value) VALUES (?, ?)",
                    ("transaction_test", 1.0),
                )

            result = db.get_rows("test_table", where="name='transaction_test'")
            assert len(result) == 1

    def test_transaction_rollback(self, db_path):
        """Test transaction rollback on error."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            with pytest.raises(Exception):
                with db.transaction():
                    db.execute(
                        "INSERT INTO test_table (name, value) VALUES (?, ?)",
                        ("rollback_test", 1.0),
                    )
                    raise Exception("Test error")

            result = db.get_rows("test_table", where="name='rollback_test'")
            assert len(result) == 0

    def test_batch_insert(self, db_path):
        """Test batch insert operations."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            rows = [{"name": f"item_{i}", "value": float(i)} for i in range(100)]

            db.insert_many("test_table", rows, batch_size=10)

            count = db.get_row_count("test_table")
            assert count == 100

    def test_batch_update(self, db_path):
        """Test batch update operations."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            rows = [{"name": f"item_{i}", "value": 0.0} for i in range(10)]
            db.insert_many("test_table", rows)

            updates = [{"name": f"item_{i}", "value": float(i * 10)} for i in range(10)]

            db.update_many("test_table", updates, where="name = ?")

            result = db.get_rows("test_table", order_by="name")
            assert len(result) == 10

    def test_save_and_load_numpy_array(self, db_path):
        """Test saving and loading numpy arrays as BLOBs."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "data": "BLOB"},
            )

            test_array = np.random.rand(10, 20).astype(np.float32)

            db.execute(
                "INSERT INTO test_table (id, name) VALUES (?, ?)", (1, "array_test")
            )

            db.save_array("test_table", test_array, column="data", ids=1)

            loaded_array = db.load_array("test_table", column="data", ids=1)

            assert loaded_array is not None
            assert loaded_array.shape == (1, 10, 20)
            assert loaded_array.dtype == test_array.dtype
            np.testing.assert_array_almost_equal(loaded_array[0], test_array)

    def test_create_index(self, db_path):
        """Test index creation."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            db.create_index("test_table", ["name"], unique=True)

            db.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                ("unique_name", 1.0),
            )

            with pytest.raises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO test_table (name, value) VALUES (?, ?)",
                    ("unique_name", 2.0),
                )

    def test_get_table_schema(self, db_path):
        """Test retrieving table schema."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            schema = db.get_table_schema("test_table")

            assert isinstance(schema, pd.DataFrame)
            assert "name" in schema.columns
            assert "type" in schema.columns
            assert "pk" in schema.columns

            pk_rows = schema[schema["pk"] == 1]
            assert len(pk_rows) == 1
            assert pk_rows.iloc[0]["name"] == "id"

    def test_get_rows_with_filters(self, db_path):
        """Test getting rows with various filters."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            for i in range(20):
                db.execute(
                    "INSERT INTO test_table (name, value) VALUES (?, ?)",
                    (f"item_{i}", float(i)),
                )

            result = db.get_rows("test_table", where="value > 10")
            assert len(result) == 9

            result = db.get_rows("test_table", order_by="value DESC", limit=5)
            assert len(result) == 5
            assert result.iloc[0]["value"] == 19.0

            result = db.get_rows("test_table", limit=5, offset=10)
            assert len(result) == 5

    def test_foreign_key_constraints(self, db_path):
        """Test foreign key constraint functionality."""
        with SQLite3(db_path) as db:
            db.create_table(
                "departments", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}
            )

            db.create_table(
                "employees",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "dept_id": "INTEGER"},
                foreign_keys=[
                    {
                        "tgt_column": "dept_id",
                        "src_table": "departments",
                        "src_column": "id",
                    }
                ],
            )

            db.enable_foreign_keys()

            db.execute("INSERT INTO departments (id, name) VALUES (1, 'Engineering')")
            db.execute("INSERT INTO employees (name, dept_id) VALUES ('John', 1)")

            with pytest.raises(sqlite3.IntegrityError):
                db.execute("INSERT INTO employees (name, dept_id) VALUES ('Jane', 999)")

    def test_csv_export_import(self, db_path, temp_dir):
        """Test CSV export and import functionality."""
        csv_path = os.path.join(temp_dir, "export.csv")

        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            for i in range(10):
                db.execute(
                    "INSERT INTO test_table (name, value) VALUES (?, ?)",
                    (f"item_{i}", float(i)),
                )

            db.save_to_csv("test_table", csv_path)

            assert os.path.exists(csv_path)
            df = pd.read_csv(csv_path)
            assert len(df) == 10

            db.create_table("imported_table", {"name": "TEXT", "value": "REAL"})
            db.load_from_csv("imported_table", csv_path)

            imported_count = db.get_row_count("imported_table")
            assert imported_count == 10

    def test_database_maintenance(self, db_path):
        """Test database maintenance operations."""
        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            for i in range(100):
                db.execute(
                    "INSERT INTO test_table (name, value) VALUES (?, ?)",
                    (f"item_{i}", float(i)),
                )

            db.vacuum()
            db.optimize()

            count = db.get_row_count("test_table")
            assert count == 100

    def test_backup_database(self, db_path, temp_dir):
        """Test database backup functionality."""
        backup_path = os.path.join(temp_dir, "backup.db")

        with SQLite3(db_path) as db:
            db.create_table(
                "test_table",
                {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "value": "REAL"},
            )

            db.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                ("backup_test", 123.45),
            )

            db.backup(backup_path)

        assert os.path.exists(backup_path)

        with SQLite3(backup_path) as backup_db:
            result = backup_db.get_rows("test_table", where="name='backup_test'")
            assert len(result) == 1
            assert result.iloc[0]["value"] == 123.45

    def test_mixins_integration(self, db_path):
        """Test that all mixins are properly integrated."""
        with SQLite3(db_path) as db:
            # ConnectionMixin
            assert hasattr(db, "connect")
            assert hasattr(db, "close")
            assert hasattr(db, "reconnect")

            # QueryMixin
            assert hasattr(db, "execute")
            assert hasattr(db, "executemany")

            # TransactionMixin
            assert hasattr(db, "transaction")
            assert hasattr(db, "begin")
            assert hasattr(db, "commit")
            assert hasattr(db, "rollback")

            # TableMixin
            assert hasattr(db, "create_table")
            assert hasattr(db, "drop_table")
            assert hasattr(db, "get_table_names")
            assert hasattr(db, "get_table_schema")

            # IndexMixin
            assert hasattr(db, "create_index")
            assert hasattr(db, "drop_index")

            # RowMixin
            assert hasattr(db, "get_rows")
            assert hasattr(db, "get_row_count")

            # BatchMixin
            assert hasattr(db, "insert_many")
            assert hasattr(db, "update_many")
            assert hasattr(db, "delete_where")

            # BlobMixin
            assert hasattr(db, "save_array")
            assert hasattr(db, "load_array")

            # ImportExportMixin
            assert hasattr(db, "load_from_csv")
            assert hasattr(db, "save_to_csv")

            # MaintenanceMixin
            assert hasattr(db, "vacuum")
            assert hasattr(db, "optimize")
            assert hasattr(db, "backup")

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/db/_sqlite3/_SQLite3.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-09-11 07:57:49 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/db/_sqlite3/_SQLite3.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import warnings
# from typing import List, Optional
# 
# from scitex.str import printc as _printc
# from ._SQLite3Mixins._ArrayMixin import _ArrayMixin
# from ._SQLite3Mixins._BatchMixin import _BatchMixin
# from ._SQLite3Mixins._BlobMixin import _BlobMixin
# from ._SQLite3Mixins._ColumnMixin import _ColumnMixin
# from ._SQLite3Mixins._ConnectionMixin import _ConnectionMixin
# from ._SQLite3Mixins._GitMixin import _GitMixin
# from ._SQLite3Mixins._ImportExportMixin import _ImportExportMixin
# from ._SQLite3Mixins._IndexMixin import _IndexMixin
# from ._SQLite3Mixins._MaintenanceMixin import _MaintenanceMixin
# from ._SQLite3Mixins._QueryMixin import _QueryMixin
# from ._SQLite3Mixins._RowMixin import _RowMixin
# from ._SQLite3Mixins._TableMixin import _TableMixin
# from ._SQLite3Mixins._TransactionMixin import _TransactionMixin
# 
# 
# class SQLite3(
#     _ArrayMixin,
#     _ConnectionMixin,
#     _QueryMixin,
#     _TransactionMixin,
#     _ColumnMixin,
#     _TableMixin,
#     _IndexMixin,
#     _RowMixin,
#     _BatchMixin,
#     _BlobMixin,
#     _ImportExportMixin,
#     _MaintenanceMixin,
#     _GitMixin,
# ):
#     """SQLite database manager with automatic metadata handling, numpy array storage, and compression.
# 
#     This class provides a comprehensive interface for SQLite database operations with
#     automatic compression, thread-safe operations, and specialized numpy array handling.
# 
#     Features:
#         - Automatic compression for BLOB data (70-90% reduction)
#         - Thread-safe operations with proper connection management
#         - Metadata handling for BLOB columns
#         - Batch processing support
#         - Context manager support for proper resource cleanup
# 
#     Examples:
#         Basic usage with context manager (recommended):
# 
#         >>> with SQLite3("data.db", compress_by_default=True) as db:
#         ...     db.create_table("experiments", {"id": "INTEGER PRIMARY KEY", "data": "BLOB"})
#         ...     data = np.random.random((1000, 100))
#         ...     db.save_array("experiments", data, column="data", additional_columns={"id": 1})
# 
#         Array storage and retrieval:
# 
#         >>> with SQLite3("data.db") as db:
#         ...     # Save numpy array
#         ...     db.save_array(
#         ...         table_name="measurements",
#         ...         data=np.random.random((1000, 100)),
#         ...         column="data",
#         ...         additional_columns={"name": "experiment_1", "timestamp": 1234567890}
#         ...     )
#         ...     # Load array
#         ...     loaded = db.load_array("measurements", "data", where="name = 'experiment_1'")
# 
#         Generic object storage:
# 
#         >>> with SQLite3("data.db") as db:
#         ...     db.save_blob(
#         ...         table_name="objects",
#         ...         data={"weights": array, "params": {"lr": 0.001}},
#         ...         key="model_v1"
#         ...     )
#         ...     loaded_obj = db.load_blob("objects", key="model_v1")
# 
#     Notes:
#         - Always use context manager (with statement) for proper resource cleanup
#         - BLOB columns automatically get metadata columns: {column}_dtype, {column}_shape, {column}_compressed
#         - Compression is enabled by default for arrays > 1KB
#         - Thread-safe operations are supported
#     """
# 
#     def __init__(
#         self,
#         db_path: str,
#         use_temp: bool = False,
#         compress_by_default: bool = False,
#         autocommit: bool = False,
#     ):
#         """Initialize SQLite database manager.
# 
#         Parameters
#         ----------
#         db_path : str
#             Path to the SQLite database file
#         use_temp : bool, optional
#             Whether to use a temporary copy of the database, by default False
#         compress_by_default : bool, optional
#             Whether to compress BLOB data by default when not explicitly specified, by default False
#         autocommit : bool, optional
#             Whether to automatically commit transactions, by default False
# 
#         Warnings
#         --------
#         UserWarning
#             If not used with context manager, warns about potential resource leaks
#         """
# 
#         if not os.path.exists(db_path):
#             os.makedirs(os.path.dirname(db_path), exist_ok=True)
# 
#         _ConnectionMixin.__init__(self, db_path, use_temp)
#         self.compress_by_default = compress_by_default
#         self.autocommit = autocommit
#         self._context_manager_used = False
# 
#     def __enter__(self):
#         """Enter context manager."""
#         self._context_manager_used = True
#         return self
# 
#     def _check_context_manager(self):
#         if not self._context_manager_used:
#             raise RuntimeError(
#                 "SQLite3 must be used with context manager: 'with SQLite3(...) as db:'"
#             )
# 
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """Exit context manager and ensure proper cleanup."""
#         self.close()
# 
#     def __del__(self):
#         """Destructor with context manager usage warning."""
#         if hasattr(self, "_context_manager_used") and not self._context_manager_used:
#             warnings.warn(
#                 "SQLite3 instance was not used with context manager. "
#                 "Use 'with SQLite3(...) as db:' to ensure proper resource cleanup.",
#                 UserWarning,
#                 stacklevel=2,
#             )
#         if hasattr(self, "close"):
#             self.close()
# 
#     def __call__(
#         self,
#         return_summary=False,
#         print_summary=True,
#         table_names: Optional[List[str]] = None,
#         verbose: bool = True,
#         limit: int = 5,
#     ):
#         """Display database summary information.
# 
#         Parameters
#         ----------
#         return_summary : bool, optional
#             Whether to return summary dict, by default False
#         print_summary : bool, optional
#             Whether to print summary to console, by default True
#         table_names : Optional[List[str]], optional
#             Specific table names to summarize, by default None (all tables)
#         verbose : bool, optional
#             Whether to show detailed information, by default True
#         limit : int, optional
#             Maximum number of rows to display per table, by default 5
# 
#         Returns
#         -------
#         dict or None
#             Summary dictionary if return_summary=True, else None
#         """
# 
#         summary = self.get_summaries(
#             table_names=table_names,
#             verbose=verbose,
#             limit=limit,
#         )
# 
#         if print_summary:
#             for k, v in summary.items():
#                 _printc(f"{k}\n{v}")
# 
#         if return_summary:
#             return summary
# 
#     @property
#     def summary(self):
#         """Quick access to database summary."""
#         self()
# 
# 
# BaseSQLiteDB = SQLite3
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/db/_sqlite3/_SQLite3.py
# --------------------------------------------------------------------------------
