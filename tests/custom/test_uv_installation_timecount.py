#!/usr/bin/env python3
# Time-stamp: "2026-01-04 (ywatanabe)"
# File: ./tests/custom/test_uv_installation_timecount.py

"""
UV Installation Time Measurement Tests for SciTeX

Measures installation time for scitex with different extras using uv.

Usage:
    # Run as pytest
    pytest tests/custom/test_uv_installation_timecount.py -v -s

    # Run as standalone script
    python tests/custom/test_uv_installation_timecount.py
    python tests/custom/test_uv_installation_timecount.py --all
"""

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Module-level extras defined in pyproject.toml
# Auto-generated from config/requirements/*.txt
DEFINED_EXTRAS = [
    "ai",  # AI/ML utilities
    "audio",  # Audio/TTS
    "browser",  # Browser automation
    "capture",  # Screen capture
    "cli",  # CLI tools
    "dev",  # Development tools
    "dsp",  # Digital signal processing
    "fig",  # Figure editing
    "gen",  # Generative AI APIs
    "ml",  # Machine learning
    "msword",  # MS Word
    "nn",  # Neural networks (heavy)
    "scholar",  # Scholar/Paper management
    "torch",  # PyTorch utilities
    "web",  # Web frameworks
    "writer",  # LaTeX compilation
    "all",  # All modules
]

# Light extras for quick testing (small dependency footprint)
LIGHT_EXTRAS = ["msword", "writer", "capture"]


@dataclass
class InstallResult:
    """Result of an installation attempt."""

    extra: str
    elapsed_time: float
    success: bool
    return_code: int
    stdout: str = ""
    stderr: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class UvInstallationTimer:
    """Measures installation time using uv."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.results: List[InstallResult] = []

    def check_uv_available(self) -> bool:
        """Check if uv is available."""
        try:
            result = subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def create_venv(self, venv_path: Path) -> bool:
        """Create a virtual environment using uv."""
        try:
            result = subprocess.run(
                ["uv", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Failed to create venv: {e}")
            return False

    def get_python_executable(self, venv_path: Path) -> Path:
        """Get python executable path for a venv."""
        if sys.platform == "win32":
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"

    def install_extra(
        self,
        extra: Optional[str] = None,
        venv_path: Optional[Path] = None,
        timeout: int = 600,
        no_cache: bool = False,
    ) -> InstallResult:
        """Install scitex with uv and measure time.

        Args:
            extra: The extra to install (e.g., 'audio', 'scholar')
            venv_path: Path to virtual environment
            timeout: Timeout in seconds
            no_cache: If True, use --no-cache for accurate cold-start timing
        """
        package_path = str(self.project_root)

        if venv_path:
            python_exe = self.get_python_executable(venv_path)
            cmd = ["uv", "pip", "install", "--python", str(python_exe)]
        else:
            cmd = ["uv", "pip", "install"]

        if no_cache:
            cmd.append("--no-cache")

        if extra:
            cmd.append(f"{package_path}[{extra}]")
        else:
            cmd.append(package_path)

        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root,
            )
            elapsed = time.perf_counter() - start_time

            install_result = InstallResult(
                extra=extra or "base",
                elapsed_time=elapsed,
                success=result.returncode == 0,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start_time
            install_result = InstallResult(
                extra=extra or "base",
                elapsed_time=elapsed,
                success=False,
                return_code=-1,
                stderr=f"Timeout after {timeout}s",
            )
        except FileNotFoundError:
            install_result = InstallResult(
                extra=extra or "base",
                elapsed_time=0,
                success=False,
                return_code=-1,
                stderr="uv not found",
            )

        self.results.append(install_result)
        return install_result

    def generate_report(self) -> str:
        """Generate a markdown report."""
        lines = [
            "# UV Installation Time Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Results",
            "",
            "| Extra | Time (s) | Status |",
            "|-------|----------|--------|",
        ]

        for result in sorted(self.results, key=lambda r: r.extra):
            status = "OK" if result.success else "FAILED"
            lines.append(f"| {result.extra} | {result.elapsed_time:.2f} | {status} |")

        # Summary stats
        successful = [r for r in self.results if r.success]
        if successful:
            avg_time = sum(r.elapsed_time for r in successful) / len(successful)
            total_time = sum(r.elapsed_time for r in self.results)
            lines.extend(
                [
                    "",
                    "## Summary",
                    f"- Total extras tested: {len(self.results)}",
                    f"- Successful: {len(successful)}",
                    f"- Failed: {len(self.results) - len(successful)}",
                    f"- Average time (successful): {avg_time:.2f}s",
                    f"- Total time: {total_time:.2f}s",
                ]
            )

        return "\n".join(lines)

    def save_results_json(self, filepath: Path) -> None:
        """Save results to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "results": [
                {
                    "extra": r.extra,
                    "elapsed_time": r.elapsed_time,
                    "success": r.success,
                    "return_code": r.return_code,
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def timer():
    """Create timer instance."""
    return UvInstallationTimer(PROJECT_ROOT)


@pytest.fixture(scope="function")
def temp_venv():
    """Create temporary venv."""
    with tempfile.TemporaryDirectory(prefix="scitex_uv_test_") as tmpdir:
        venv_path = Path(tmpdir) / "venv"
        yield venv_path


@pytest.fixture(scope="module")
def uv_available():
    """Check if uv is available."""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.benchmark
class TestUvInstallationTime:
    """Test uv installation times."""

    @pytest.mark.parametrize("extra", DEFINED_EXTRAS)
    def test_uv_install_extra(self, timer, temp_venv, uv_available, extra):
        """Test uv installation time for each extra."""
        if not uv_available:
            pytest.skip("uv not available")

        timer.create_venv(temp_venv)
        result = timer.install_extra(extra=extra, venv_path=temp_venv, timeout=600)

        print(
            f"\nuv pip install scitex[{extra}]: {result.elapsed_time:.2f}s - {'OK' if result.success else 'FAILED'}"
        )

        assert result.elapsed_time > 0

    def test_uv_install_base(self, timer, temp_venv, uv_available):
        """Test uv installation time for base package."""
        if not uv_available:
            pytest.skip("uv not available")

        timer.create_venv(temp_venv)
        result = timer.install_extra(extra=None, venv_path=temp_venv, timeout=300)

        print(
            f"\nuv pip install scitex (base): {result.elapsed_time:.2f}s - {'OK' if result.success else 'FAILED'}"
        )

        assert result.elapsed_time > 0


# =============================================================================
# Standalone execution
# =============================================================================


def run_benchmark(
    extras: Optional[List[str]] = None,
    save_report: bool = True,
    no_cache: bool = False,
):
    """Run full benchmark.

    Args:
        extras: List of extras to test. If None, uses DEFINED_EXTRAS.
        save_report: Whether to save reports.
        no_cache: If True, use --no-cache for accurate cold-start timing.
    """
    if extras is None:
        extras = DEFINED_EXTRAS

    timer = UvInstallationTimer(PROJECT_ROOT)

    if not timer.check_uv_available():
        print("ERROR: uv is not available")
        sys.exit(1)

    cache_mode = "disabled (--no-cache)" if no_cache else "enabled"
    print("UV Installation Time Benchmark")
    print("=" * 60)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Extras to test: {len(extras)}")
    print(f"Cache: {cache_mode}")
    print("=" * 60)

    # Test base package first
    print("\nTesting: base (no extras)")
    with tempfile.TemporaryDirectory(prefix="uv_base_") as tmpdir:
        venv_path = Path(tmpdir) / "venv"
        if timer.create_venv(venv_path):
            result = timer.install_extra(
                extra=None, venv_path=venv_path, no_cache=no_cache
            )
            status = "OK" if result.success else "FAILED"
            print(f"  uv: {result.elapsed_time:.2f}s - {status}")

    # Test each extra
    for extra in extras:
        print(f"\nTesting: {extra}")
        with tempfile.TemporaryDirectory(prefix=f"uv_{extra}_") as tmpdir:
            venv_path = Path(tmpdir) / "venv"
            if timer.create_venv(venv_path):
                result = timer.install_extra(
                    extra=extra, venv_path=venv_path, no_cache=no_cache
                )
                status = "OK" if result.success else "FAILED"
                print(f"  uv: {result.elapsed_time:.2f}s - {status}")

    # Generate reports
    if save_report:
        report_dir = PROJECT_ROOT / "tests" / "custom" / "reports"
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_path = report_dir / f"uv_install_times_{timestamp}.json"
        timer.save_results_json(json_path)
        print(f"\nJSON saved: {json_path}")

        # Save Markdown
        md_path = report_dir / f"uv_install_times_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(timer.generate_report())
        print(f"Report saved: {md_path}")

    print("\n" + timer.generate_report())

    return timer.results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UV installation time benchmark")
    parser.add_argument(
        "--extras",
        nargs="*",
        default=None,
        help="Extras to test (default: all defined extras)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all defined extras",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Test only light extras (msword, writer, audio)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save reports",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable uv cache for accurate cold-start timing (slower but realistic)",
    )
    args = parser.parse_args()

    if args.light:
        extras = LIGHT_EXTRAS
    elif args.extras:
        extras = args.extras
    else:
        extras = DEFINED_EXTRAS

    run_benchmark(extras=extras, save_report=not args.no_save, no_cache=args.no_cache)

# EOF
