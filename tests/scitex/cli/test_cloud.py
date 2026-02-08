#!/usr/bin/env python3
"""Tests for scitex.cli.cloud - Cloud CLI delegation to scitex-cloud."""

import subprocess

import pytest
from click.testing import CliRunner

from scitex.cli.cloud import HAS_CLOUD, cloud


class TestCloudCommand:
    """Tests for the cloud command delegation."""

    def test_cloud_help(self):
        """Test that cloud help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["--help"])
        assert result.exit_code == 0
        assert "Cloud" in result.output or "cloud" in result.output

    def test_cloud_short_help(self):
        """Test that -h also shows help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["-h"])
        assert result.exit_code == 0

    @pytest.mark.skipif(not HAS_CLOUD, reason="scitex-cloud not installed")
    def test_cloud_delegates_to_scitex_cloud(self):
        """Test that cloud command delegates to scitex-cloud CLI."""
        # Test that scitex-cloud is callable
        result = subprocess.run(
            ["scitex-cloud", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "scitex-cloud" in result.stdout

    @pytest.mark.skipif(not HAS_CLOUD, reason="scitex-cloud not installed")
    def test_cloud_gitea_help(self):
        """Test that cloud gitea subcommand works."""
        result = subprocess.run(
            ["scitex-cloud", "gitea", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "clone" in result.stdout
        assert "create" in result.stdout

    @pytest.mark.skipif(not HAS_CLOUD, reason="scitex-cloud not installed")
    def test_cloud_mcp_help(self):
        """Test that cloud mcp subcommand works."""
        result = subprocess.run(
            ["scitex-cloud", "mcp", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "start" in result.stdout
        assert "doctor" in result.stdout
        assert "list-tools" in result.stdout

    @pytest.mark.skipif(not HAS_CLOUD, reason="scitex-cloud not installed")
    def test_cloud_mcp_list_tools(self):
        """Test that mcp list-tools shows available tools."""
        result = subprocess.run(
            ["scitex-cloud", "mcp", "list-tools"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "cloud" in result.stdout.lower() or "api" in result.stdout.lower()


# EOF
