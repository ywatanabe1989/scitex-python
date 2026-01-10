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
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cli/resource.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# """
# SciTeX CLI - Resource Commands (System Monitoring)
# 
# Provides system resource monitoring and specifications.
# """
# 
# import sys
# 
# import click
# 
# 
# @click.group(context_settings={"help_option_names": ["-h", "--help"]})
# def resource():
#     """
#     System resource monitoring
# 
#     \b
#     Commands:
#       specs     Show system specifications
#       usage     Show current resource usage
#       monitor   Continuously monitor resource usage
# 
#     \b
#     Examples:
#       scitex resource specs              # Show system specs
#       scitex resource usage              # Current CPU/memory/GPU usage
#       scitex resource monitor --interval 5
#     """
#     pass
# 
# 
# @resource.command()
# @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
# @click.option(
#     "--category",
#     "-c",
#     multiple=True,
#     type=click.Choice(["cpu", "memory", "disk", "network", "gpu", "os", "python"]),
#     help="Specific category to show",
# )
# def specs(as_json, category):
#     """
#     Show system specifications
# 
#     \b
#     Categories:
#       cpu     - Processor information
#       memory  - RAM information
#       disk    - Storage information
#       network - Network interfaces
#       gpu     - GPU/CUDA information
#       os      - Operating system details
#       python  - Python environment
# 
#     \b
#     Examples:
#       scitex resource specs
#       scitex resource specs --json
#       scitex resource specs --category cpu --category gpu
#     """
#     try:
#         from scitex.resource import get_specs
# 
#         specs_data = get_specs()
# 
#         # Filter categories if specified
#         if category:
#             category_map = {
#                 "cpu": "_cpu_info",
#                 "memory": "_memory_info",
#                 "disk": "_disk_info",
#                 "network": "_network_info",
#                 "gpu": "_supple_nvidia_info",
#                 "os": "_supple_os_info",
#                 "python": "_supple_python_info",
#             }
#             filtered = {}
#             for cat in category:
#                 key = category_map.get(cat, cat)
#                 if key in specs_data:
#                     filtered[key] = specs_data[key]
#                 elif cat in specs_data:
#                     filtered[cat] = specs_data[cat]
#             specs_data = filtered
# 
#         if as_json:
#             import json
# 
#             click.echo(json.dumps(specs_data, indent=2, default=str))
#         else:
#             click.secho("System Specifications", fg="cyan", bold=True)
#             click.echo("=" * 50)
# 
#             for section, data in specs_data.items():
#                 section_name = (
#                     section.replace("_info", "").replace("_supple_", "").upper()
#                 )
#                 click.secho(f"\n{section_name}:", fg="yellow")
#                 if isinstance(data, dict):
#                     for key, value in data.items():
#                         click.echo(f"  {key}: {value}")
#                 else:
#                     click.echo(f"  {data}")
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @resource.command()
# @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
# def usage(as_json):
#     """
#     Show current resource usage (CPU, memory, GPU)
# 
#     \b
#     Examples:
#       scitex resource usage
#       scitex resource usage --json
#     """
#     try:
#         from scitex.resource import get_processor_usages
# 
#         usage_data = get_processor_usages()
# 
#         if as_json:
#             import json
# 
#             click.echo(json.dumps(usage_data, indent=2, default=str))
#         else:
#             click.secho("Resource Usage", fg="cyan", bold=True)
#             click.echo("=" * 50)
# 
#             # CPU
#             cpu = usage_data.get("cpu", {})
#             click.secho("\nCPU:", fg="yellow")
#             click.echo(f"  Usage: {cpu.get('percent', 'N/A')}%")
#             click.echo(f"  Cores: {cpu.get('count', 'N/A')}")
# 
#             # Memory
#             mem = usage_data.get("memory", {})
#             click.secho("\nMemory:", fg="yellow")
#             click.echo(f"  Used: {mem.get('percent', 'N/A')}%")
#             click.echo(f"  Total: {mem.get('total_gb', 'N/A')} GB")
#             click.echo(f"  Available: {mem.get('available_gb', 'N/A')} GB")
# 
#             # GPU (if available)
#             gpu = usage_data.get("gpu", {})
#             if gpu:
#                 click.secho("\nGPU:", fg="yellow")
#                 for i, g in enumerate(gpu.get("devices", [])):
#                     click.echo(f"  [{i}] {g.get('name', 'Unknown')}")
#                     click.echo(
#                         f"      Memory: {g.get('memory_used', 'N/A')} / {g.get('memory_total', 'N/A')} MB"
#                     )
#                     click.echo(f"      Utilization: {g.get('utilization', 'N/A')}%")
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @resource.command()
# @click.option(
#     "--interval",
#     "-i",
#     type=float,
#     default=2.0,
#     help="Update interval in seconds (default: 2.0)",
# )
# @click.option("--count", "-n", type=int, help="Number of updates (default: continuous)")
# @click.option("--log", "-l", type=click.Path(), help="Log to file")
# def monitor(interval, count, log):
#     """
#     Continuously monitor resource usage
# 
#     \b
#     Examples:
#       scitex resource monitor
#       scitex resource monitor --interval 5
#       scitex resource monitor --count 10 --log usage.log
#     """
#     try:
#         import time
# 
#         from scitex.resource import get_processor_usages
# 
#         click.echo(f"Monitoring resources (interval: {interval}s)")
#         click.echo("Press Ctrl+C to stop")
#         click.echo()
# 
#         log_file = None
#         if log:
#             log_file = open(log, "w")
#             log_file.write("timestamp,cpu_percent,memory_percent,gpu_percent\n")
# 
#         iteration = 0
#         try:
#             while True:
#                 if count and iteration >= count:
#                     break
# 
#                 usage_data = get_processor_usages()
#                 cpu_pct = usage_data.get("cpu", {}).get("percent", 0)
#                 mem_pct = usage_data.get("memory", {}).get("percent", 0)
#                 gpu_pct = 0
#                 gpu_info = usage_data.get("gpu", {})
#                 if gpu_info and gpu_info.get("devices"):
#                     gpu_pct = gpu_info["devices"][0].get("utilization", 0)
# 
#                 # Display
#                 from datetime import datetime
# 
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 line = f"[{ts}] CPU: {cpu_pct:5.1f}%  MEM: {mem_pct:5.1f}%  GPU: {gpu_pct:5.1f}%"
#                 click.echo(line)
# 
#                 # Log
#                 if log_file:
#                     log_file.write(f"{ts},{cpu_pct},{mem_pct},{gpu_pct}\n")
#                     log_file.flush()
# 
#                 iteration += 1
#                 time.sleep(interval)
# 
#         except KeyboardInterrupt:
#             click.echo("\nMonitoring stopped")
#         finally:
#             if log_file:
#                 log_file.close()
#                 click.echo(f"Log saved: {log}")
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# if __name__ == "__main__":
#     resource()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cli/resource.py
# --------------------------------------------------------------------------------
