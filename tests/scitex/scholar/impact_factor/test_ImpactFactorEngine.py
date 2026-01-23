#!/usr/bin/env python3
# Timestamp: "2026-01-14"
# File: tests/scitex/scholar/impact_factor/test_ImpactFactorEngine.py
"""
Comprehensive tests for ImpactFactorEngine.

Tests cover:
- Initialization with cache settings
- Journal metrics lookup
- JCR year extraction
- Database info retrieval
- Standalone get_journal_metrics function
- Cache behavior with lru_cache
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.impact_factor import ImpactFactorEngine, get_journal_metrics


class TestImpactFactorEngineInit:
    """Tests for ImpactFactorEngine initialization."""

    def test_init_default_cache_size(self):
        """Engine should initialize with default cache size."""
        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine.ImpactFactorJCREngine"
        ) as mock_jcr:
            mock_jcr.return_value = MagicMock()

            engine = ImpactFactorEngine()

            assert engine.name == "ImpactFactorEngine"
            assert hasattr(engine, "get_metrics")

    def test_init_custom_cache_size(self):
        """Engine should accept custom cache size."""
        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine.ImpactFactorJCREngine"
        ) as mock_jcr:
            mock_jcr.return_value = MagicMock()

            engine = ImpactFactorEngine(cache_size=500)

            # Cache is set up via lru_cache
            assert engine.get_metrics is not None


class TestImpactFactorEngineGetMetrics:
    """Tests for get_metrics functionality."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked JCR database."""
        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine.ImpactFactorJCREngine"
        ) as mock_jcr_class:
            mock_jcr = MagicMock()
            mock_jcr.search.return_value = [{"factor": 42.778, "jcr": "Q1"}]
            mock_jcr.dbfile = "/path/to/jcr_2023.db"
            mock_jcr_class.return_value = mock_jcr

            engine = ImpactFactorEngine()
            engine.jcr_engine = mock_jcr
            return engine

    def test_get_metrics_returns_data(self, mock_engine):
        """Should return metrics for known journal."""
        with patch.object(mock_engine, "_get_jcr_year", return_value="JCR 2023"):
            # Call the uncached method directly
            result = mock_engine._get_metrics_uncached("Nature")

        assert result is not None
        assert result["impact_factor"] == 42.778
        assert result["quartile"] == "Q1"
        assert "JCR" in result["source"]

    def test_get_metrics_empty_journal(self, mock_engine):
        """Should return None for empty journal name."""
        result = mock_engine._get_metrics_uncached("")

        assert result is None

    def test_get_metrics_none_journal(self, mock_engine):
        """Should return None for None journal name."""
        result = mock_engine._get_metrics_uncached(None)

        assert result is None

    def test_get_metrics_not_found(self, mock_engine):
        """Should return None when journal not in database."""
        mock_engine.jcr_engine.search.return_value = []

        result = mock_engine._get_metrics_uncached("Unknown Journal XYZ")

        assert result is None

    def test_get_metrics_exception_handling(self, mock_engine):
        """Should return None on exception."""
        mock_engine.jcr_engine.search.side_effect = Exception("Database error")

        result = mock_engine._get_metrics_uncached("Nature")

        assert result is None


class TestImpactFactorEngineJCRYear:
    """Tests for JCR year extraction."""

    @pytest.fixture
    def engine_with_db(self, tmp_path):
        """Create engine with mock database file."""
        db_path = tmp_path / "jcr_2023.db"

        # Create actual SQLite database
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE journals (name TEXT, factor REAL, jcr TEXT)")
            cursor.execute("INSERT INTO journals VALUES ('Nature', 42.778, 'Q1')")
            conn.commit()

        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine.ImpactFactorJCREngine"
        ) as mock_jcr_class:
            mock_jcr = MagicMock()
            mock_jcr.dbfile = db_path
            mock_jcr_class.return_value = mock_jcr

            engine = ImpactFactorEngine()
            engine.jcr_engine = mock_jcr
            return engine

    def test_get_jcr_year_from_filename(self, engine_with_db):
        """Should extract year from database filename."""
        result = engine_with_db._get_jcr_year()

        assert "2023" in result

    def test_get_jcr_year_unknown(self, tmp_path):
        """Should return 'Source Unknown' when no year found."""
        db_path = tmp_path / "database.db"

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE data (id INTEGER)")
            conn.commit()

        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine.ImpactFactorJCREngine"
        ) as mock_jcr_class:
            mock_jcr = MagicMock()
            mock_jcr.dbfile = db_path
            mock_jcr_class.return_value = mock_jcr

            engine = ImpactFactorEngine()
            engine.jcr_engine = mock_jcr

            result = engine._get_jcr_year()

            assert result == "Source Unknown"


class TestImpactFactorEngineGetDatabaseInfo:
    """Tests for get_database_info functionality."""

    @pytest.fixture
    def engine_with_full_db(self, tmp_path):
        """Create engine with fully populated database."""
        db_path = tmp_path / "jcr_2023.db"

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE journals (name TEXT, factor REAL, jcr TEXT)")
            cursor.execute("INSERT INTO journals VALUES ('Nature', 42.778, 'Q1')")
            cursor.execute("INSERT INTO journals VALUES ('Science', 41.845, 'Q1')")
            cursor.execute("INSERT INTO journals VALUES ('Cell', 38.637, 'Q1')")
            conn.commit()

        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine.ImpactFactorJCREngine"
        ) as mock_jcr_class:
            mock_jcr = MagicMock()
            mock_jcr.dbfile = db_path
            mock_jcr_class.return_value = mock_jcr

            engine = ImpactFactorEngine()
            engine.jcr_engine = mock_jcr
            return engine

    def test_get_database_info_returns_dict(self, engine_with_full_db):
        """Should return database info dictionary."""
        result = engine_with_full_db.get_database_info()

        assert "database_path" in result
        assert "tables" in result
        assert "total_journals" in result
        assert result["total_journals"] == 3

    def test_get_database_info_includes_columns(self, engine_with_full_db):
        """Should include column names."""
        result = engine_with_full_db.get_database_info()

        assert "columns" in result
        assert "name" in result["columns"]
        assert "factor" in result["columns"]

    def test_get_database_info_includes_sample(self, engine_with_full_db):
        """Should include sample data."""
        result = engine_with_full_db.get_database_info()

        assert "sample_data" in result
        assert len(result["sample_data"]) == 3

    def test_get_database_info_no_jcr_engine(self):
        """Should return error when no JCR engine."""
        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine.ImpactFactorJCREngine"
        ) as mock_jcr_class:
            mock_jcr_class.return_value = None

            engine = ImpactFactorEngine()
            engine.jcr_engine = None

            result = engine.get_database_info()

            assert "error" in result


class TestGetJournalMetricsFunction:
    """Tests for standalone get_journal_metrics function."""

    def test_get_journal_metrics_returns_none_for_unknown(self):
        """Function should return None for unknown journal."""
        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine.ImpactFactorJCREngine"
        ) as mock_jcr_class:
            mock_jcr = MagicMock()
            mock_jcr.search.return_value = []  # No results
            mock_jcr.dbfile = "/path/to/jcr_2023.db"
            mock_jcr_class.return_value = mock_jcr

            result = get_journal_metrics("Unknown Journal XYZ 12345")

            assert result is None

    def test_get_journal_metrics_returns_metrics_for_known(self):
        """Function should return metrics for known journal."""
        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine.ImpactFactorJCREngine"
        ) as mock_jcr_class:
            mock_jcr = MagicMock()
            mock_jcr.search.return_value = [{"factor": 42.778, "jcr": "Q1"}]
            mock_jcr.dbfile = "/path/to/jcr_2023.db"
            mock_jcr_class.return_value = mock_jcr

            result = get_journal_metrics("Nature")

            # Must return a result with expected structure
            assert result is not None
            assert "impact_factor" in result
            assert result["impact_factor"] == 42.778
            assert "quartile" in result
            assert result["quartile"] == "Q1"


class TestImpactFactorEngineCaching:
    """Tests for caching behavior."""

    def test_cache_returns_same_result(self):
        """Should return cached result for repeated calls."""
        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine.ImpactFactorJCREngine"
        ) as mock_jcr_class:
            mock_jcr = MagicMock()
            mock_jcr.search.return_value = [{"factor": 42.778, "jcr": "Q1"}]
            mock_jcr.dbfile = "/path/to/jcr_2023.db"
            mock_jcr_class.return_value = mock_jcr

            engine = ImpactFactorEngine()
            engine.jcr_engine = mock_jcr

            with patch.object(engine, "_get_jcr_year", return_value="JCR 2023"):
                # First call
                result1 = engine.get_metrics("Nature")
                # Second call (should be cached)
                result2 = engine.get_metrics("Nature")

            # Both should return same result
            assert result1 == result2

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/impact_factor/ImpactFactorEngine.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-11 23:58:17 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/extra/JournalMetrics.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/scholar/extra/JournalMetrics.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# __FILE__ = __file__
# 
# """
# Functionalities:
# - Retrieves journal impact factors and quartiles
# - Provides standalone journal metrics lookup
# - Caches results for performance optimization
# 
# Dependencies:
# - packages:
#   - impact_factor
# 
# Input:
# - Journal names as strings
# 
# Output:
# - Dictionary containing impact factor and quartile data
# """
# 
# """Imports"""
# from functools import lru_cache
# from typing import Dict, Optional
# 
# from .jcr.ImpactFactorJCREngine import ImpactFactorJCREngine
# 
# """Parameters"""
# 
# """Functions & Classes"""
# 
# 
# class ImpactFactorEngine:
#     """
#     Impact factor service - finds journal metrics from JCR database.
# 
#     Uses JCR database lookup with caching for performance.
#     """
# 
#     def __init__(self, cache_size: int = 1000):
#         """Initialize with optional cache size."""
#         self.name = self.__class__.__name__
#         self.jcr_engine = ImpactFactorJCREngine()
#         self.get_metrics = lru_cache(maxsize=cache_size)(self._get_metrics_uncached)
# 
#     def _get_jcr_year(self) -> str:
#         """Extract JCR year from database or package metadata."""
#         try:
#             import sqlite3
# 
#             with sqlite3.connect(self.jcr_engine.dbfile) as conn:
#                 cursor = conn.cursor()
# 
#                 # Check if there's a metadata table with year info
#                 cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
#                 tables = [row[0] for row in cursor.fetchall()]
# 
#                 if "metadata" in tables:
#                     cursor.execute(
#                         "SELECT value FROM metadata WHERE key='year' OR key='jcr_year'"
#                     )
#                     year_result = cursor.fetchone()
#                     if year_result:
#                         return f"JCR {year_result[0]}"
# 
#                 # Try to extract year from database filename
#                 try:
#                     import re
# 
#                     db_path = str(self.jcr_engine.dbfile)
#                     year_match = re.search(r"20\d{2}", db_path)
#                     if year_match:
#                         return f"JCR {year_match.group()}"
#                 except:
#                     pass
# 
#         except Exception:
#             pass
# 
#         return "Source Unknown"
# 
#     def _get_metrics_uncached(self, journal_name: str) -> Optional[Dict]:
#         """Get journal metrics without caching."""
#         if not self.jcr_engine or not journal_name:
#             return None
# 
#         try:
#             results = self.jcr_engine.search(journal_name)
#             if results:
#                 result = results[0]
#                 return {
#                     "impact_factor": float(result.get("factor", 0)),
#                     "quartile": result.get("jcr", "Unknown"),
#                     "source": self._get_jcr_year(),
#                 }
#         except Exception:
#             pass
# 
#         return None
# 
#     def get_database_info(self) -> Dict:
#         """Get information about the impact factor database."""
#         if not self.jcr_engine:
#             return {"error": "Database not available"}
# 
#         import sqlite3
# 
#         db_path = self.jcr_engine.dbfile
# 
#         with sqlite3.connect(db_path) as conn:
#             cursor = conn.cursor()
# 
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
#             tables = [row[0] for row in cursor.fetchall()]
# 
#             info = {
#                 "database_path": str(db_path),
#                 "tables": tables,
#                 "total_journals": 0,
#                 "data_year": self._get_jcr_year(),
#             }
# 
#             if tables:
#                 main_table = tables[0]
#                 cursor.execute(f"SELECT COUNT(*) FROM {main_table}")
#                 info["total_journals"] = cursor.fetchone()[0]
# 
#                 cursor.execute(f"PRAGMA table_info({main_table})")
#                 columns = [row[1] for row in cursor.fetchall()]
#                 info["columns"] = columns
# 
#                 cursor.execute(f"SELECT * FROM {main_table} LIMIT 3")
#                 sample_data = cursor.fetchall()
#                 info["sample_data"] = sample_data
# 
#             return info
# 
# 
# def get_journal_metrics(journal_name: str) -> Optional[Dict]:
#     """Standalone function to get journal metrics.
# 
#     Parameters
#     ----------
#     journal_name : str
#         Name of the journal
# 
#     Returns
#     -------
#     Optional[Dict]
#         Dictionary with impact_factor, quartile, and source keys
# 
#     Example
#     -------
#     >>> metrics = get_journal_metrics("Nature")
#     >>> print(metrics["impact_factor"])
#     64.8
#     """
#     engine = ImpactFactorEngine()
#     return engine.get_metrics(journal_name)
# 
# 
# if __name__ == "__main__":
# 
#     def main():
#         """Demonstrate journal metrics lookup."""
#         metrics_instance = ImpactFactorEngine()
# 
#         # Show database info
#         print("Database Information")
#         print("=" * 50)
#         db_info = metrics_instance.get_database_info()
#         for key, value in db_info.items():
#             print(f"{key}: {value}")
# 
#         print("\nJournal Metrics Lookup Demo")
#         print("=" * 50)
# 
#         test_journals = ["Nature", "Science", "Cell"]
# 
#         for journal in test_journals:
#             print(f"\nJournal: {journal}")
#             metrics = get_journal_metrics(journal)
#             if metrics:
#                 for key, value in metrics.items():
#                     print(f"  {key}: {value}")
#             else:
#                 print("  No metrics found")
# 
#     main()
# # python -m scitex.scholar.extra.JournalMetrics
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/impact_factor/ImpactFactorEngine.py
# --------------------------------------------------------------------------------
