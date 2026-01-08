#!/usr/bin/env python3
"""Tests for scitex.cli.repro - Reproducibility CLI commands."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.repro import repro


class TestReproGroup:
    """Tests for the repro command group."""

    def test_repro_help(self):
        """Test that repro help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(repro, ["--help"])
        assert result.exit_code == 0
        assert "Reproducibility utilities" in result.output

    def test_repro_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(repro, ["--help"])
        expected_commands = ["gen-id", "gen-timestamp", "hash", "seed"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in repro help"


class TestReproGenId:
    """Tests for the repro gen-id command."""

    def test_gen_id_default(self):
        """Test gen-id with default options."""
        runner = CliRunner()
        with patch("scitex.repro.gen_ID") as mock_gen:
            mock_gen.return_value = "abc12345"
            result = runner.invoke(repro, ["gen-id"])
            assert result.exit_code == 0
            assert "abc12345" in result.output

    def test_gen_id_with_length(self):
        """Test gen-id with custom length."""
        runner = CliRunner()
        with patch("scitex.repro.gen_ID") as mock_gen:
            mock_gen.return_value = "abcdefghijkl"
            result = runner.invoke(repro, ["gen-id", "--length", "4"])
            assert result.exit_code == 0
            # Output should be truncated to 4 chars
            assert len(result.output.strip()) == 4

    def test_gen_id_with_prefix(self):
        """Test gen-id with prefix."""
        runner = CliRunner()
        with patch("scitex.repro.gen_ID") as mock_gen:
            mock_gen.return_value = "abc12345"
            result = runner.invoke(repro, ["gen-id", "--prefix", "exp_"])
            assert result.exit_code == 0
            assert result.output.strip().startswith("exp_")

    def test_gen_id_multiple(self):
        """Test gen-id with count option."""
        runner = CliRunner()
        with patch("scitex.repro.gen_ID") as mock_gen:
            mock_gen.side_effect = ["id1_____", "id2_____", "id3_____"]
            result = runner.invoke(repro, ["gen-id", "--count", "3"])
            assert result.exit_code == 0
            lines = result.output.strip().split("\n")
            assert len(lines) == 3


class TestReproGenTimestamp:
    """Tests for the repro gen-timestamp command."""

    def test_gen_timestamp_default(self):
        """Test gen-timestamp with default ISO format."""
        runner = CliRunner()
        result = runner.invoke(repro, ["gen-timestamp"])
        assert result.exit_code == 0
        # ISO format contains T separator
        assert "T" in result.output or "-" in result.output

    def test_gen_timestamp_file_format(self):
        """Test gen-timestamp with file-safe format."""
        runner = CliRunner()
        result = runner.invoke(repro, ["gen-timestamp", "--format", "file"])
        assert result.exit_code == 0
        # File format should be like 20250108_123045
        output = result.output.strip()
        assert "_" in output
        assert len(output) == 15  # YYYYMMDD_HHMMSS

    def test_gen_timestamp_compact_format(self):
        """Test gen-timestamp with compact format."""
        runner = CliRunner()
        result = runner.invoke(repro, ["gen-timestamp", "--format", "compact"])
        assert result.exit_code == 0
        output = result.output.strip()
        assert len(output) == 14  # YYYYMMDDHHMMSS
        assert "_" not in output

    def test_gen_timestamp_human_format(self):
        """Test gen-timestamp with human-readable format."""
        runner = CliRunner()
        result = runner.invoke(repro, ["gen-timestamp", "--format", "human"])
        assert result.exit_code == 0
        # Human format like "Jan 08, 2025 12:30:45"
        assert "," in result.output or ":" in result.output

    def test_gen_timestamp_utc(self):
        """Test gen-timestamp with UTC flag."""
        runner = CliRunner()
        result = runner.invoke(repro, ["gen-timestamp", "--utc"])
        assert result.exit_code == 0
        # Should produce valid timestamp
        assert len(result.output.strip()) > 0


class TestReproHash:
    """Tests for the repro hash command."""

    def test_hash_file(self):
        """Test hash command on a regular file."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            result = runner.invoke(repro, ["hash", temp_path])
            assert result.exit_code == 0
            # Should output hash and filename
            assert temp_path.split("/")[-1] in result.output
            # Hash should be 64 chars for sha256
            parts = result.output.strip().split()
            assert len(parts[0]) == 64
        finally:
            os.unlink(temp_path)

    def test_hash_short(self):
        """Test hash command with --short flag."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            result = runner.invoke(repro, ["hash", temp_path, "--short"])
            assert result.exit_code == 0
            parts = result.output.strip().split()
            # Short hash should be 8 chars
            assert len(parts[0]) == 8
        finally:
            os.unlink(temp_path)

    def test_hash_nonexistent_file(self):
        """Test hash command on non-existent file."""
        runner = CliRunner()
        result = runner.invoke(repro, ["hash", "/nonexistent/file.txt"])
        assert result.exit_code != 0


class TestReproSeed:
    """Tests for the repro seed command."""

    def test_seed_basic(self):
        """Test seed command."""
        runner = CliRunner()
        with patch("scitex.repro.RandomStateManager") as mock_rsm:
            result = runner.invoke(repro, ["seed", "42"])
            assert result.exit_code == 0
            assert "Random seed set to: 42" in result.output
            mock_rsm.assert_called_once_with(seed=42, verbose=False)

    def test_seed_verbose(self):
        """Test seed command with verbose flag."""
        runner = CliRunner()
        with patch("scitex.repro.RandomStateManager") as mock_rsm:
            result = runner.invoke(repro, ["seed", "12345", "--verbose"])
            assert result.exit_code == 0
            assert "Seeded libraries" in result.output
            mock_rsm.assert_called_once_with(seed=12345, verbose=True)

    def test_seed_error_handling(self):
        """Test seed command handles errors."""
        runner = CliRunner()
        with patch("scitex.repro.RandomStateManager") as mock_rsm:
            mock_rsm.side_effect = Exception("Seed failed")
            result = runner.invoke(repro, ["seed", "42"])
            assert result.exit_code == 1
            assert "Error" in result.output


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
