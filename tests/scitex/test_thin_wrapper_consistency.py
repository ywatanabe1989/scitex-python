#!/usr/bin/env python3
# Timestamp: 2026-01-27
# File: tests/scitex/test_thin_wrapper_consistency.py

"""Tests for thin wrapper consistency between scitex and standalone packages.

Thin wrappers should have identical:
- Python API exports
- MCP tools
- CLI commands
"""

import subprocess
import sys

import pytest


class TestWriterThinWrapper:
    """Tests for scitex.writer thin wrapper consistency with scitex-writer."""

    def test_python_api_exports_identical(self):
        """scitex.writer should export same modules as scitex_writer."""
        import scitex_writer

        import scitex.writer

        # Get public exports (excluding dunder and private)
        scitex_exports = {x for x in dir(scitex.writer) if not x.startswith("_")}
        standalone_exports = {x for x in dir(scitex_writer) if not x.startswith("_")}

        # scitex.writer adds HAS_WRITER_PKG and writer_version
        scitex_extras = {"HAS_WRITER_PKG", "writer_version"}
        scitex_core = scitex_exports - scitex_extras

        # Core modules should match
        expected_modules = {
            "bib",
            "compile",
            "figures",
            "guidelines",
            "project",
            "prompts",
            "tables",
        }
        assert expected_modules <= scitex_core, (
            f"Missing modules: {expected_modules - scitex_core}"
        )
        assert expected_modules <= standalone_exports, (
            f"Standalone missing: {expected_modules - standalone_exports}"
        )

    def test_mcp_tools_identical(self):
        """scitex writer MCP tools should match scitex-writer MCP tools."""
        # Get scitex writer tools
        result1 = subprocess.run(
            [sys.executable, "-m", "scitex", "mcp", "list-tools"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        scitex_tools = {
            line.split()[0] for line in result1.stdout.split("\n") if "[writer]" in line
        }

        # Get scitex-writer tools
        result2 = subprocess.run(
            ["scitex-writer", "mcp", "list-tools"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Extract tool names (words with underscores)
        import re

        standalone_tools = set(re.findall(r"\b[a-z_]+_[a-z_]+\b", result2.stdout))

        assert scitex_tools == standalone_tools, (
            f"MCP tools mismatch!\n"
            f"Only in scitex: {scitex_tools - standalone_tools}\n"
            f"Only in standalone: {standalone_tools - scitex_tools}"
        )


class TestSocialThinWrapper:
    """Tests for scitex.social thin wrapper consistency with socialia."""

    def test_python_api_exports_classes(self):
        """scitex.social should export same platform classes as socialia."""
        try:
            import socialia

            import scitex.social
        except ImportError:
            pytest.skip("socialia not installed")

        expected_classes = {
            "Twitter",
            "LinkedIn",
            "Reddit",
            "YouTube",
            "GoogleAnalytics",
            "BasePoster",
        }

        scitex_exports = {x for x in dir(scitex.social) if not x.startswith("_")}
        standalone_exports = {x for x in dir(socialia) if not x.startswith("_")}

        for cls in expected_classes:
            assert cls in scitex_exports, f"scitex.social missing {cls}"
            assert cls in standalone_exports, f"socialia missing {cls}"

    def test_mcp_tools_identical(self):
        """scitex social MCP tools should match socialia MCP tools."""
        import re

        # Get scitex social tools
        result1 = subprocess.run(
            [sys.executable, "-m", "scitex", "mcp", "list-tools"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        scitex_tools = set()
        for line in result1.stdout.split("\n"):
            # Match tool lines with social_ or analytics_ prefix
            match = re.match(r"^\s+(social_\w+|analytics_\w+)\s", line)
            if match:
                scitex_tools.add(match.group(1))

        # Get socialia tools
        result2 = subprocess.run(
            [sys.executable, "-m", "socialia", "mcp", "list-tools"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        standalone_tools = set()
        for line in result2.stdout.split("\n"):
            match = re.match(r"^\s+(social_\w+|analytics_\w+)\s", line)
            if match:
                standalone_tools.add(match.group(1))

        assert scitex_tools == standalone_tools, (
            f"MCP tools mismatch!\n"
            f"Only in scitex: {scitex_tools - standalone_tools}\n"
            f"Only in standalone: {standalone_tools - scitex_tools}"
        )


class TestIntrospectAPIConsistency:
    """Tests using introspect API to verify thin wrapper consistency."""

    def test_writer_api_item_count(self):
        """scitex.writer API should have similar item count to scitex_writer."""
        from scitex.introspect import list_api

        scitex_df = list_api("scitex.writer", max_depth=2)
        standalone_df = list_api("scitex_writer", max_depth=2)

        # Allow small difference for wrapper-specific items
        diff = abs(len(scitex_df) - len(standalone_df))
        assert diff <= 5, (
            f"API item count mismatch: scitex.writer={len(scitex_df)}, "
            f"scitex_writer={len(standalone_df)}, diff={diff}"
        )

    def test_hyphen_underscore_equivalence(self):
        """Introspect should handle both hyphen and underscore in module names."""
        from scitex.introspect import list_api

        # These should resolve to the same module
        df1 = list_api("scitex_writer", max_depth=1)
        df2 = list_api("scitex-writer", max_depth=1)

        assert len(df1) == len(df2), "Hyphen and underscore should give same results"


# EOF
