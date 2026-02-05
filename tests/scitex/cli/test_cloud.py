#!/usr/bin/env python3
"""Tests for scitex.cli.cloud - Cloud CLI delegation to scitex-cloud."""

import pytest
from click.testing import CliRunner

from scitex.cli.cloud import HAS_CLOUD, cloud


class TestCloudGroup:
    """Tests for the cloud command group."""

    def test_cloud_help(self):
        """Test that cloud help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["--help"])
        assert result.exit_code == 0
        assert "Cloud" in result.output

    def test_cloud_short_help(self):
        """Test that -h also shows help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["-h"])
        assert result.exit_code == 0

    @pytest.mark.skipif(not HAS_CLOUD, reason="scitex-cloud not installed")
    def test_cloud_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["--help"])
        expected_commands = [
            "login",
            "clone",
            "create",
            "list",
            "search",
            "delete",
            "fork",
            "pr",
            "issue",
            "push",
            "pull",
            "status",
            "enrich",
        ]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found"


@pytest.mark.skipif(not HAS_CLOUD, reason="scitex-cloud not installed")
class TestCloudDelegation:
    """Tests for cloud CLI delegation to scitex-cloud."""

    def test_login_help(self):
        """Test login command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["login", "--help"])
        assert result.exit_code == 0

    def test_clone_help(self):
        """Test clone command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["clone", "--help"])
        assert result.exit_code == 0
        assert "Clone" in result.output

    def test_create_help(self):
        """Test create command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["create", "--help"])
        assert result.exit_code == 0

    def test_list_help(self):
        """Test list command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["list", "--help"])
        assert result.exit_code == 0

    def test_search_help(self):
        """Test search command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["search", "--help"])
        assert result.exit_code == 0

    def test_pr_help(self):
        """Test PR command group help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["pr", "--help"])
        assert result.exit_code == 0

    def test_issue_help(self):
        """Test issue command group help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["issue", "--help"])
        assert result.exit_code == 0

    def test_push_help(self):
        """Test push command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["push", "--help"])
        assert result.exit_code == 0

    def test_pull_help(self):
        """Test pull command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["pull", "--help"])
        assert result.exit_code == 0

    def test_status_help(self):
        """Test status command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["status", "--help"])
        assert result.exit_code == 0

    def test_enrich_help(self):
        """Test enrich command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["enrich", "--help"])
        assert result.exit_code == 0


# EOF
