#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/compile.py

"""
LaTeX compilation functions with Python wrappers.

This module provides Python wrappers around scitex-writer shell scripts,
handling subprocess execution, output parsing, and exit code management.
"""

import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CompilationResult:
    """Result of LaTeX compilation."""

    success: bool
    """Whether compilation succeeded (exit code 0)"""

    exit_code: int
    """Process exit code"""

    stdout: str
    """Standard output from compilation"""

    stderr: str
    """Standard error from compilation"""

    output_pdf: Optional[Path] = None
    """Path to generated PDF (if successful)"""

    diff_pdf: Optional[Path] = None
    """Path to diff PDF with tracked changes (if generated)"""

    log_file: Optional[Path] = None
    """Path to compilation log file"""

    duration: float = 0.0
    """Compilation duration in seconds"""

    errors: List[str] = None
    """Parsed LaTeX errors (if any)"""

    warnings: List[str] = None
    """Parsed LaTeX warnings (if any)"""

    def __post_init__(self):
        """Initialize mutable fields."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

    def __str__(self):
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Compilation {status} (exit code: {self.exit_code})",
            f"Duration: {self.duration:.2f}s",
        ]
        if self.output_pdf:
            lines.append(f"Output: {self.output_pdf}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        return "\n".join(lines)


def _find_compile_script() -> Path:
    """
    Find the compile script from scitex-writer.

    Returns:
        Path to compile script

    Raises:
        FileNotFoundError: If compile script not found
    """
    # Try common locations
    locations = [
        Path("/tmp/scitex-writer/compile"),
        Path.home() / "proj" / "scitex-writer" / "compile",
        Path(__file__).parent / "scripts" / "compile",
    ]

    for location in locations:
        if location.exists():
            return location

    raise FileNotFoundError(
        "scitex-writer compile script not found. "
        "Please clone scitex-writer to /tmp/scitex-writer or ~/proj/scitex-writer"
    )


def _run_compile(
    doc_type: str,
    project_dir: Path,
    additional_args: List[str] = None,
    timeout: int = 300
) -> CompilationResult:
    """
    Run compilation script and parse results.

    Args:
        doc_type: Document type ('manuscript', 'supplementary', 'revision')
        project_dir: Path to project directory (containing 01_manuscript/, etc.)
        additional_args: Additional arguments to pass to compile script
        timeout: Timeout in seconds

    Returns:
        CompilationResult with compilation status and outputs
    """
    start_time = datetime.now()

    try:
        compile_script = _find_compile_script()
    except FileNotFoundError as e:
        logger.error(str(e))
        return CompilationResult(
            success=False,
            exit_code=127,
            stdout="",
            stderr=str(e),
            duration=0.0
        )

    # Build command
    cmd = [str(compile_script), f"-{doc_type[0]}"]  # -m, -s, or -r
    if additional_args:
        cmd.extend(additional_args)

    logger.info(f"Running compilation: {' '.join(cmd)}")
    logger.info(f"Working directory: {project_dir}")

    try:
        # Run compilation
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Determine output paths
        doc_map = {
            'manuscript': '01_manuscript',
            'supplementary': '02_supplementary',
            'revision': '03_revision'
        }
        doc_dir = project_dir / doc_map[doc_type]

        output_pdf = None
        diff_pdf = None
        log_file = None

        if result.returncode == 0:
            # Find generated PDF
            pdf_name = f"{doc_type}.pdf"
            potential_pdf = doc_dir / pdf_name
            if potential_pdf.exists():
                output_pdf = potential_pdf

            # Check for diff PDF
            diff_name = f"{doc_type}_diff.pdf"
            potential_diff = doc_dir / diff_name
            if potential_diff.exists():
                diff_pdf = potential_diff

            # Find log file
            log_dir = doc_dir / "logs"
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                if log_files:
                    log_file = max(log_files, key=lambda p: p.stat().st_mtime)

        # Parse errors and warnings from output
        errors = _parse_latex_errors(result.stdout + result.stderr)
        warnings = _parse_latex_warnings(result.stdout + result.stderr)

        compilation_result = CompilationResult(
            success=(result.returncode == 0),
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            output_pdf=output_pdf,
            diff_pdf=diff_pdf,
            log_file=log_file,
            duration=duration,
            errors=errors,
            warnings=warnings
        )

        if compilation_result.success:
            logger.info(f"Compilation succeeded in {duration:.2f}s")
            if output_pdf:
                logger.info(f"Output PDF: {output_pdf}")
        else:
            logger.error(f"Compilation failed with exit code {result.returncode}")
            if errors:
                logger.error(f"Found {len(errors)} errors")

        return compilation_result

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Compilation timed out after {duration:.2f}s")
        return CompilationResult(
            success=False,
            exit_code=124,  # Timeout exit code
            stdout="",
            stderr=f"Compilation timed out after {timeout}s",
            duration=duration
        )

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Compilation error: {e}")
        return CompilationResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr=str(e),
            duration=duration
        )


def _parse_latex_errors(output: str) -> List[str]:
    """
    Parse LaTeX errors from compilation output.

    Args:
        output: Compilation output (stdout + stderr)

    Returns:
        List of error messages
    """
    errors = []
    lines = output.split('\n')

    for i, line in enumerate(lines):
        # LaTeX error pattern: "! Error: ..."
        if line.startswith('!'):
            errors.append(line.strip())
            # Include next line if it provides context
            if i + 1 < len(lines) and lines[i + 1].strip():
                errors.append(lines[i + 1].strip())

    return errors


def _parse_latex_warnings(output: str) -> List[str]:
    """
    Parse LaTeX warnings from compilation output.

    Args:
        output: Compilation output (stdout + stderr)

    Returns:
        List of warning messages
    """
    warnings = []
    for line in output.split('\n'):
        # LaTeX warning patterns
        if 'Warning:' in line or 'warning:' in line.lower():
            warnings.append(line.strip())

    return warnings


def compile_manuscript(
    project_dir: Path,
    additional_args: List[str] = None,
    timeout: int = 300
) -> CompilationResult:
    """
    Compile manuscript document.

    Args:
        project_dir: Path to writer project directory (containing 01_manuscript/)
        additional_args: Additional arguments for compilation
        timeout: Timeout in seconds

    Returns:
        CompilationResult with compilation status and outputs

    Examples:
        >>> from pathlib import Path
        >>> result = compile_manuscript(Path("/path/to/project"))
        >>> if result.success:
        ...     print(f"PDF created: {result.output_pdf}")
        ... else:
        ...     print(f"Errors: {result.errors}")
    """
    return _run_compile('manuscript', project_dir, additional_args, timeout)


def compile_supplementary(
    project_dir: Path,
    additional_args: List[str] = None,
    timeout: int = 300
) -> CompilationResult:
    """
    Compile supplementary materials.

    Args:
        project_dir: Path to writer project directory (containing 02_supplementary/)
        additional_args: Additional arguments for compilation
        timeout: Timeout in seconds

    Returns:
        CompilationResult with compilation status and outputs
    """
    return _run_compile('supplementary', project_dir, additional_args, timeout)


def compile_revision(
    project_dir: Path,
    track_changes: bool = False,
    additional_args: List[str] = None,
    timeout: int = 300
) -> CompilationResult:
    """
    Compile revision responses.

    Args:
        project_dir: Path to writer project directory (containing 03_revision/)
        track_changes: Whether to enable change tracking
        additional_args: Additional arguments for compilation
        timeout: Timeout in seconds

    Returns:
        CompilationResult with compilation status and outputs
    """
    args = additional_args or []
    if track_changes:
        args.append('--track-changes')

    return _run_compile('revision', project_dir, args, timeout)


__all__ = [
    'CompilationResult',
    'compile_manuscript',
    'compile_supplementary',
    'compile_revision',
]

# EOF
