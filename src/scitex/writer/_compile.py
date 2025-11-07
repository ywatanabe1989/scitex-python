#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time stamp: "2025-11-07 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/_compile.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
LaTeX compilation functions with live log streaming support.

This module provides Python wrappers around scitex-writer shell scripts,
handling subprocess execution, output parsing, exit code management, and
optional live log/progress callbacks for real-time updates.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
import subprocess
import time
import fcntl

from ._validate_tree_structures import validate_tree_structures
from .utils._parse_latex_logs import parse_compilation_output
from .dataclasses.config import DOC_TYPE_DIRS, DOC_TYPE_FLAGS
from .dataclasses import CompilationResult

from scitex.logging import getLogger

logger = getLogger(__name__)


def _execute_with_callbacks(
    command: list,
    cwd: Path,
    timeout: int,
    log_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Execute command with line-by-line output capture and callbacks.

    Args:
        command: Command to execute as list
        cwd: Working directory
        timeout: Timeout in seconds
        log_callback: Called with each output line

    Returns:
        Dict with stdout, stderr, exit_code, success
    """
    # Set environment for unbuffered output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    process = subprocess.Popen(
        command,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,  # Unbuffered
        cwd=str(cwd),
        env=env,
    )

    stdout_lines = []
    stderr_lines = []
    start_time = time.time()

    # Make file descriptors non-blocking
    def make_non_blocking(fd):
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    make_non_blocking(process.stdout)
    make_non_blocking(process.stderr)

    stdout_buffer = b""
    stderr_buffer = b""

    try:
        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                process.kill()
                timeout_msg = f"[ERROR] Command timed out after {timeout} seconds"
                if log_callback:
                    log_callback(timeout_msg)
                stderr_lines.append(timeout_msg)
                break

            # Check if process has finished
            poll_result = process.poll()

            # Read from stdout
            try:
                chunk = process.stdout.read()
                if chunk:
                    stdout_buffer += chunk
                    # Process complete lines
                    while b'\n' in stdout_buffer:
                        line, stdout_buffer = stdout_buffer.split(b'\n', 1)
                        line_str = line.decode("utf-8", errors="replace")
                        stdout_lines.append(line_str)
                        if log_callback:
                            log_callback(line_str)
            except (IOError, BlockingIOError):
                pass

            # Read from stderr
            try:
                chunk = process.stderr.read()
                if chunk:
                    stderr_buffer += chunk
                    # Process complete lines
                    while b'\n' in stderr_buffer:
                        line, stderr_buffer = stderr_buffer.split(b'\n', 1)
                        line_str = line.decode("utf-8", errors="replace")
                        stderr_lines.append(line_str)
                        if log_callback:
                            log_callback(f"[STDERR] {line_str}")
            except (IOError, BlockingIOError):
                pass

            # If process finished, do final read and break
            if poll_result is not None:
                # Process remaining buffer content
                if stdout_buffer:
                    line_str = stdout_buffer.decode("utf-8", errors="replace")
                    stdout_lines.append(line_str)
                    if log_callback:
                        log_callback(line_str)

                if stderr_buffer:
                    line_str = stderr_buffer.decode("utf-8", errors="replace")
                    stderr_lines.append(line_str)
                    if log_callback:
                        log_callback(f"[STDERR] {line_str}")

                break

            # Small sleep to prevent CPU spinning
            time.sleep(0.05)

    except Exception as e:
        process.kill()
        raise

    return {
        "stdout": "\n".join(stdout_lines),
        "stderr": "\n".join(stderr_lines),
        "exit_code": process.returncode,
        "success": process.returncode == 0,
    }


def _run_compile(
    doc_type: str,
    project_dir: Path,
    timeout: int = 300,
    track_changes: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> CompilationResult:
    """
    Run compilation script and parse results with optional live callbacks.

    Args:
        doc_type: Document type ('manuscript', 'supplementary', 'revision')
        project_dir: Path to project directory (containing 01_manuscript/, etc.)
        timeout: Timeout in seconds
        track_changes: Enable change tracking (revision only)
        log_callback: Called with each log line: log_callback("Running pdflatex...")
        progress_callback: Called with progress: progress_callback(50, "Pass 2/3")

    Returns:
        CompilationResult with compilation status and outputs
    """
    start_time = datetime.now()
    project_dir = Path(project_dir).absolute()

    # Helper functions for logging and progress
    full_log = []

    def log(message: str):
        """Log a message and invoke callback if provided."""
        full_log.append(message)
        logger.info(message)
        if log_callback:
            log_callback(message)

    def progress(percent: int, step: str):
        """Update progress and invoke callback if provided."""
        logger.info(f"Progress: {percent}% - {step}")
        if progress_callback:
            progress_callback(percent, step)

    # Stage 0: Preparation
    progress(0, 'Starting compilation...')
    log('[INFO] Starting LaTeX compilation...')

    # Validate project structure before compilation
    try:
        validate_tree_structures(project_dir)
        progress(5, 'Validated project structure')
        log('[INFO] Project structure validated')
    except Exception as e:
        error_msg = f"[ERROR] Project structure validation failed: {e}"
        log(error_msg)
        return CompilationResult(
            success=False,
            exit_code=1,
            stdout="\n".join(full_log),
            stderr=error_msg,
            duration=0.0,
        )

    # Get compile script from scripts/shell/ directory
    script_map = {
        "manuscript": project_dir
        / "scripts"
        / "shell"
        / "compile_manuscript.sh",
        "supplementary": project_dir
        / "scripts"
        / "shell"
        / "compile_supplementary.sh",
        "revision": project_dir / "scripts" / "shell" / "compile_revision.sh",
    }

    compile_script = script_map.get(doc_type)
    if not compile_script or not compile_script.exists():
        error_msg = f"[ERROR] Compilation script not found: {compile_script}"
        log(error_msg)
        return CompilationResult(
            success=False,
            exit_code=127,
            stdout="\n".join(full_log),
            stderr=error_msg,
            duration=0.0,
        )

    # Build command - use absolute path for script
    script_path = compile_script.absolute()
    cmd = [str(script_path)]
    if track_changes and doc_type == "revision":
        cmd.append("--track-changes")

    progress(10, 'Preparing files and environment...')
    log(f'[INFO] Running compilation: {" ".join(cmd)}')
    log(f'[INFO] Working directory: {project_dir}')

    try:
        # Execute with streaming and callbacks
        progress(15, 'Executing LaTeX compilation...')
        log('[INFO] Starting pdflatex process...')

        result_dict = _execute_with_callbacks(
            command=cmd,
            cwd=project_dir,
            timeout=timeout,
            log_callback=log_callback,  # Pass callback for line-by-line streaming
        )

        # Add output to our log
        if result_dict["stdout"]:
            full_log.extend(result_dict["stdout"].split('\n'))
        if result_dict["stderr"]:
            full_log.extend([f"[STDERR] {line}" for line in result_dict["stderr"].split('\n')])

        duration = (datetime.now() - start_time).total_seconds()

        # Determine output paths
        doc_dir = project_dir / DOC_TYPE_DIRS[doc_type]

        output_pdf = None
        diff_pdf = None
        log_file = None

        if result_dict["success"]:
            progress(90, 'Compilation successful, locating output files...')
            log('[INFO] Compilation succeeded, checking output files...')

            # Find generated PDF
            pdf_name = f"{doc_type}.pdf"
            potential_pdf = doc_dir / pdf_name
            if potential_pdf.exists():
                output_pdf = potential_pdf
                log(f'[SUCCESS] PDF generated: {output_pdf}')

            # Check for diff PDF
            diff_name = f"{doc_type}_diff.pdf"
            potential_diff = doc_dir / diff_name
            if potential_diff.exists():
                diff_pdf = potential_diff
                log(f'[INFO] Diff PDF generated: {diff_pdf}')

            # Find log file
            log_dir = doc_dir / "logs"
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                if log_files:
                    log_file = max(log_files, key=lambda p: p.stat().st_mtime)
                    log(f'[INFO] Log file: {log_file}')

        # Parse errors and warnings using parse_latex_logs module
        progress(95, 'Parsing LaTeX logs...')
        error_issues, warning_issues = parse_compilation_output(
            result_dict["stdout"] + result_dict["stderr"], log_file=log_file
        )

        # Convert LaTeXIssue objects to strings for backward compatibility
        errors = [str(issue) for issue in error_issues]
        warnings = [str(issue) for issue in warning_issues]

        compilation_result = CompilationResult(
            success=result_dict["success"],
            exit_code=result_dict["exit_code"],
            stdout="\n".join(full_log),
            stderr=result_dict["stderr"],
            output_pdf=output_pdf,
            diff_pdf=diff_pdf,
            log_file=log_file,
            duration=duration,
            errors=errors,
            warnings=warnings,
        )

        if compilation_result.success:
            progress(100, 'Complete!')
            log(f'[SUCCESS] Compilation succeeded in {duration:.2f}s')
            if output_pdf:
                log(f'[SUCCESS] Output PDF: {output_pdf}')
        else:
            progress(100, 'Compilation failed')
            log(f'[ERROR] Compilation failed with exit code {result_dict["exit_code"]}')
            if errors:
                log(f'[ERROR] Found {len(errors)} errors')

        return compilation_result

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"[ERROR] Compilation exception: {e}"
        log(error_msg)
        progress(100, 'Compilation failed with exception')
        return CompilationResult(
            success=False,
            exit_code=1,
            stdout="\n".join(full_log),
            stderr=str(e),
            duration=duration,
        )


def compile_manuscript(
    project_dir: Path,
    timeout: int = 300,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> CompilationResult:
    """
    Compile manuscript document with optional live callbacks.

    Args:
        project_dir: Path to writer project directory (containing 01_manuscript/)
        timeout: Timeout in seconds
        log_callback: Called with each log line: log_callback("Running pdflatex...")
        progress_callback: Called with progress: progress_callback(50, "Pass 2/3")

    Returns:
        CompilationResult with compilation status and outputs

    Examples:
        >>> from pathlib import Path
        >>> result = compile_manuscript(Path("/path/to/project"))
        >>> if result.success:
        ...     print(f"PDF created: {result.output_pdf}")
        ... else:
        ...     print(f"Errors: {result.errors}")

        >>> # With callbacks
        >>> def on_log(msg):
        ...     print(f"LOG: {msg}")
        >>> def on_progress(percent, step):
        ...     print(f"{percent}%: {step}")
        >>> result = compile_manuscript(
        ...     Path("/path/to/project"),
        ...     log_callback=on_log,
        ...     progress_callback=on_progress
        ... )
    """
    return _run_compile(
        "manuscript",
        project_dir,
        timeout=timeout,
        log_callback=log_callback,
        progress_callback=progress_callback,
    )


def compile_supplementary(
    project_dir: Path,
    timeout: int = 300,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> CompilationResult:
    """
    Compile supplementary materials with optional live callbacks.

    Args:
        project_dir: Path to writer project directory (containing 02_supplementary/)
        timeout: Timeout in seconds
        log_callback: Called with each log line
        progress_callback: Called with progress updates

    Returns:
        CompilationResult with compilation status and outputs
    """
    return _run_compile(
        "supplementary",
        project_dir,
        timeout=timeout,
        log_callback=log_callback,
        progress_callback=progress_callback,
    )


def compile_revision(
    project_dir: Path,
    track_changes: bool = False,
    timeout: int = 300,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> CompilationResult:
    """
    Compile revision responses with optional live callbacks.

    Args:
        project_dir: Path to writer project directory (containing 03_revision/)
        track_changes: Whether to enable change tracking
        timeout: Timeout in seconds
        log_callback: Called with each log line
        progress_callback: Called with progress updates

    Returns:
        CompilationResult with compilation status and outputs
    """
    return _run_compile(
        "revision",
        project_dir,
        timeout=timeout,
        track_changes=track_changes,
        log_callback=log_callback,
        progress_callback=progress_callback,
    )


__all__ = [
    "CompilationResult",
    "compile_manuscript",
    "compile_supplementary",
    "compile_revision",
]

# EOF
