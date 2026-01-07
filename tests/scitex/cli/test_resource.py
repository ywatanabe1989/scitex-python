#!/usr/bin/env python3
"""Tests for scitex.cli.resource - System resource monitoring CLI commands."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.resource import resource


class TestResourceGroup:
    """Tests for the resource command group."""

    def test_resource_help(self):
        """Test that resource help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(resource, ["--help"])
        assert result.exit_code == 0
        assert "System resource monitoring" in result.output

    def test_resource_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(resource, ["--help"])
        expected_commands = ["specs", "usage", "monitor"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in resource help"


class TestResourceSpecs:
    """Tests for the resource specs command."""

    def test_specs_default(self):
        """Test specs command with default options."""
        runner = CliRunner()
        with patch("scitex.resource.get_specs") as mock_specs:
            mock_specs.return_value = {
                "_cpu_info": {"model": "Intel i7", "cores": 8},
                "_memory_info": {"total_gb": 32},
            }
            result = runner.invoke(resource, ["specs"])
            assert result.exit_code == 0
            assert "System Specifications" in result.output
            assert "CPU" in result.output

    def test_specs_json(self):
        """Test specs command with --json flag."""
        runner = CliRunner()
        with patch("scitex.resource.get_specs") as mock_specs:
            mock_specs.return_value = {
                "_cpu_info": {"model": "Intel i7", "cores": 8},
            }
            result = runner.invoke(resource, ["specs", "--json"])
            assert result.exit_code == 0
            output = json.loads(result.output)
            assert "_cpu_info" in output

    def test_specs_filter_category(self):
        """Test specs command with category filter."""
        runner = CliRunner()
        with patch("scitex.resource.get_specs") as mock_specs:
            mock_specs.return_value = {
                "_cpu_info": {"model": "Intel i7", "cores": 8},
                "_memory_info": {"total_gb": 32},
                "_supple_nvidia_info": {"devices": []},
            }
            result = runner.invoke(resource, ["specs", "--category", "cpu"])
            assert result.exit_code == 0
            # Should only show CPU info
            assert "CPU" in result.output

    def test_specs_multiple_categories(self):
        """Test specs command with multiple category filters."""
        runner = CliRunner()
        with patch("scitex.resource.get_specs") as mock_specs:
            mock_specs.return_value = {
                "_cpu_info": {"model": "Intel i7"},
                "_memory_info": {"total_gb": 32},
                "_supple_nvidia_info": {"devices": []},
            }
            result = runner.invoke(
                resource, ["specs", "--category", "cpu", "--category", "memory"]
            )
            assert result.exit_code == 0

    def test_specs_error_handling(self):
        """Test specs command handles errors."""
        runner = CliRunner()
        with patch("scitex.resource.get_specs") as mock_specs:
            mock_specs.side_effect = Exception("Failed to get specs")
            result = runner.invoke(resource, ["specs"])
            assert result.exit_code == 1
            assert "Error" in result.output


class TestResourceUsage:
    """Tests for the resource usage command."""

    def test_usage_default(self):
        """Test usage command with default options."""
        runner = CliRunner()
        with patch("scitex.resource.get_processor_usages") as mock_usage:
            mock_usage.return_value = {
                "cpu": {"percent": 25.5, "count": 8},
                "memory": {"percent": 45.2, "total_gb": 32, "available_gb": 17.5},
                "gpu": {},
            }
            result = runner.invoke(resource, ["usage"])
            assert result.exit_code == 0
            assert "Resource Usage" in result.output
            assert "CPU" in result.output
            assert "Memory" in result.output

    def test_usage_json(self):
        """Test usage command with --json flag."""
        runner = CliRunner()
        with patch("scitex.resource.get_processor_usages") as mock_usage:
            mock_usage.return_value = {
                "cpu": {"percent": 25.5},
                "memory": {"percent": 45.2},
            }
            result = runner.invoke(resource, ["usage", "--json"])
            assert result.exit_code == 0
            output = json.loads(result.output)
            assert "cpu" in output
            assert "memory" in output

    def test_usage_with_gpu(self):
        """Test usage command with GPU information."""
        runner = CliRunner()
        with patch("scitex.resource.get_processor_usages") as mock_usage:
            mock_usage.return_value = {
                "cpu": {"percent": 25.5, "count": 8},
                "memory": {"percent": 45.2, "total_gb": 32, "available_gb": 17.5},
                "gpu": {
                    "devices": [
                        {
                            "name": "RTX 3090",
                            "memory_used": 4000,
                            "memory_total": 24000,
                            "utilization": 30,
                        }
                    ]
                },
            }
            result = runner.invoke(resource, ["usage"])
            assert result.exit_code == 0
            assert "GPU" in result.output
            assert "RTX 3090" in result.output

    def test_usage_error_handling(self):
        """Test usage command handles errors."""
        runner = CliRunner()
        with patch("scitex.resource.get_processor_usages") as mock_usage:
            mock_usage.side_effect = Exception("Failed to get usage")
            result = runner.invoke(resource, ["usage"])
            assert result.exit_code == 1
            assert "Error" in result.output


class TestResourceMonitor:
    """Tests for the resource monitor command."""

    def test_monitor_with_count(self):
        """Test monitor command with limited iterations."""
        runner = CliRunner()
        with patch("scitex.resource.get_processor_usages") as mock_usage:
            with patch("time.sleep"):  # Don't actually sleep
                mock_usage.return_value = {
                    "cpu": {"percent": 25.5},
                    "memory": {"percent": 45.2},
                    "gpu": {},
                }
                result = runner.invoke(resource, ["monitor", "--count", "2"])
                assert result.exit_code == 0
                assert "Monitoring resources" in result.output

    def test_monitor_with_interval(self):
        """Test monitor command with custom interval."""
        runner = CliRunner()
        with patch("scitex.resource.get_processor_usages") as mock_usage:
            with patch("time.sleep"):
                mock_usage.return_value = {
                    "cpu": {"percent": 25.5},
                    "memory": {"percent": 45.2},
                    "gpu": {},
                }
                result = runner.invoke(
                    resource, ["monitor", "--interval", "5", "--count", "1"]
                )
                assert result.exit_code == 0
                # Interval should be used
                assert "interval: 5" in result.output

    def test_monitor_keyboard_interrupt(self):
        """Test monitor command handles keyboard interrupt."""
        runner = CliRunner()
        with patch("scitex.resource.get_processor_usages") as mock_usage:
            with patch("time.sleep") as mock_sleep:
                mock_usage.return_value = {
                    "cpu": {"percent": 25.5},
                    "memory": {"percent": 45.2},
                    "gpu": {},
                }
                # Simulate KeyboardInterrupt on second call
                mock_sleep.side_effect = [None, KeyboardInterrupt()]
                result = runner.invoke(resource, ["monitor"])
                assert result.exit_code == 0
                assert "stopped" in result.output.lower()


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
