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
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cli/repro.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# """
# SciTeX CLI - Repro Commands (Reproducibility)
# 
# Provides reproducibility utilities: ID generation, timestamps, hashing.
# """
# 
# import sys
# 
# import click
# 
# 
# @click.group(context_settings={"help_option_names": ["-h", "--help"]})
# def repro():
#     """
#     Reproducibility utilities
# 
#     \b
#     Commands:
#       gen-id         Generate unique identifier
#       gen-timestamp  Generate timestamp
#       hash           Hash array/file for reproducibility
#       seed           Set random seed across all libraries
# 
#     \b
#     Examples:
#       scitex repro gen-id              # Generate unique ID
#       scitex repro gen-timestamp       # Generate timestamp
#       scitex repro hash data.npy       # Hash array file
#       scitex repro seed 42             # Set random seed
#     """
#     pass
# 
# 
# @repro.command("gen-id")
# @click.option("--length", "-l", type=int, default=8, help="ID length (default: 8)")
# @click.option("--prefix", "-p", default="", help="Prefix to add to ID")
# @click.option("--count", "-n", type=int, default=1, help="Number of IDs to generate")
# def gen_id(length, prefix, count):
#     """
#     Generate unique identifier(s)
# 
#     \b
#     Examples:
#       scitex repro gen-id
#       scitex repro gen-id --length 12
#       scitex repro gen-id --prefix exp_
#       scitex repro gen-id --count 5
#     """
#     try:
#         from scitex.repro import gen_ID
# 
#         for _ in range(count):
#             id_str = gen_ID()
#             if length != 8:
#                 id_str = id_str[:length]
#             if prefix:
#                 id_str = f"{prefix}{id_str}"
#             click.echo(id_str)
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @repro.command("gen-timestamp")
# @click.option(
#     "--format",
#     "-f",
#     "fmt",
#     type=click.Choice(["iso", "file", "compact", "human"]),
#     default="iso",
#     help="Timestamp format (default: iso)",
# )
# @click.option("--utc", is_flag=True, help="Use UTC timezone")
# def gen_timestamp(fmt, utc):
#     """
#     Generate timestamp
# 
#     \b
#     Formats:
#       iso     - ISO 8601 format (2025-01-08T12:30:45)
#       file    - File-safe format (20250108_123045)
#       compact - Compact format (20250108123045)
#       human   - Human readable (Jan 08, 2025 12:30:45)
# 
#     \b
#     Examples:
#       scitex repro gen-timestamp
#       scitex repro gen-timestamp --format file
#       scitex repro gen-timestamp --format human --utc
#     """
#     try:
#         from datetime import datetime, timezone
# 
#         from scitex.repro import gen_timestamp as make_timestamp
# 
#         if utc:
#             now = datetime.now(timezone.utc)
#         else:
#             now = datetime.now()
# 
#         if fmt == "iso":
#             ts = now.isoformat()
#         elif fmt == "file":
#             ts = now.strftime("%Y%m%d_%H%M%S")
#         elif fmt == "compact":
#             ts = now.strftime("%Y%m%d%H%M%S")
#         elif fmt == "human":
#             ts = now.strftime("%b %d, %Y %H:%M:%S")
#         else:
#             ts = make_timestamp()
# 
#         click.echo(ts)
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @repro.command()
# @click.argument("file_path", type=click.Path(exists=True))
# @click.option(
#     "--algorithm", "-a", default="sha256", help="Hash algorithm (default: sha256)"
# )
# @click.option("--short", "-s", is_flag=True, help="Output short hash (first 8 chars)")
# def hash(file_path, algorithm, short):
#     """
#     Hash array or file for reproducibility verification
# 
#     \b
#     Supported file types:
#       .npy, .npz  - NumPy arrays
#       .pt, .pth   - PyTorch tensors
#       .pkl        - Pickle files
#       *           - Any file (raw bytes hash)
# 
#     \b
#     Examples:
#       scitex repro hash data.npy
#       scitex repro hash model.pt --short
#       scitex repro hash weights.npz --algorithm md5
#     """
#     try:
#         import hashlib
#         from pathlib import Path
# 
#         path = Path(file_path)
# 
#         # Try to load as array first
#         hash_val = None
#         try:
#             if path.suffix in (".npy", ".npz"):
#                 import numpy as np
# 
#                 from scitex.repro import hash_array
# 
#                 arr = np.load(path, allow_pickle=True)
#                 if isinstance(arr, np.lib.npyio.NpzFile):
#                     # For npz, hash all arrays
#                     hashes = []
#                     for key in arr.files:
#                         hashes.append(hash_array(arr[key]))
#                     hash_val = hashlib.sha256("".join(hashes).encode()).hexdigest()
#                 else:
#                     hash_val = hash_array(arr)
#             elif path.suffix in (".pt", ".pth"):
#                 import torch
# 
#                 data = torch.load(path, map_location="cpu")
#                 if isinstance(data, torch.Tensor):
#                     hash_val = hashlib.sha256(data.numpy().tobytes()).hexdigest()
#                 else:
#                     # State dict or other
#                     hash_val = hashlib.sha256(str(data).encode()).hexdigest()
#         except ImportError:
#             pass
# 
#         # Fall back to file hash
#         if hash_val is None:
#             h = hashlib.new(algorithm)
#             with open(path, "rb") as f:
#                 for chunk in iter(lambda: f.read(8192), b""):
#                     h.update(chunk)
#             hash_val = h.hexdigest()
# 
#         if short:
#             hash_val = hash_val[:8]
# 
#         click.echo(f"{hash_val}  {path.name}")
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @repro.command()
# @click.argument("seed", type=int)
# @click.option("--verbose", "-v", is_flag=True, help="Show what was seeded")
# def seed(seed, verbose):
#     """
#     Set random seed across all available libraries
# 
#     \b
#     Affects: os, random, numpy, torch, tensorflow, jax
# 
#     \b
#     Examples:
#       scitex repro seed 42
#       scitex repro seed 12345 --verbose
#     """
#     try:
#         from scitex.repro import RandomStateManager
# 
#         rsm = RandomStateManager(seed=seed, verbose=verbose)
# 
#         click.secho(f"Random seed set to: {seed}", fg="green")
#         if verbose:
#             click.echo("Seeded libraries:")
#             click.echo("  - os.environ['PYTHONHASHSEED']")
#             click.echo("  - random")
#             click.echo("  - numpy (if available)")
#             click.echo("  - torch (if available)")
#             click.echo("  - tensorflow (if available)")
#             click.echo("  - jax (if available)")
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# if __name__ == "__main__":
#     repro()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cli/repro.py
# --------------------------------------------------------------------------------
