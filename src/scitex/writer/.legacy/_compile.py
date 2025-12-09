#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 05:43:24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_compile.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
LaTeX compilation functions with Python wrappers.

This module provides Python wrappers around scitex-writer shell scripts,
handling subprocess execution, output parsing, and exit code management.
"""

from pathlib import Path
from datetime import datetime

from ._validate_tree_structures import validate_tree_structures
from .utils._parse_latex_logs import parse_compilation_output
from .dataclasses.config import DOC_TYPE_DIRS, DOC_TYPE_FLAGS
from .dataclasses import CompilationResult

from scitex.logging import getLogger
from scitex._sh import sh

logger = getLogger(__name__)


def _run_compile(
    doc_type: str,
    project_dir: Path,
    timeout: int = 300,
    track_changes: bool = False,
) -> CompilationResult:
    """
    Run compilation script and parse results.

    Args:
        doc_type: Document type ('manuscript', 'supplementary', 'revision')
        project_dir: Path to project directory (containing 01_manuscript/, etc.)
        timeout: Timeout in seconds
        track_changes: Enable change tracking (revision only)

    Returns:
        CompilationResult with compilation status and outputs
    """
    start_time = datetime.now()
    project_dir = Path(project_dir).absolute()

    # Validate project structure before compilation
    validate_tree_structures(project_dir)

    # Get compile script from scripts/shell/ directory
    script_map = {
        "manuscript": project_dir / "scripts" / "shell" / "compile_manuscript.sh",
        "supplementary": project_dir / "scripts" / "shell" / "compile_supplementary.sh",
        "revision": project_dir / "scripts" / "shell" / "compile_revision.sh",
    }

    compile_script = script_map.get(doc_type)
    if not compile_script or not compile_script.exists():
        error_msg = f"Compilation script not found: {compile_script}"
        logger.error(error_msg)
        return CompilationResult(
            success=False,
            exit_code=127,
            stdout="",
            stderr=error_msg,
            duration=0.0,
        )

    # Build command - use absolute path for script
    script_path = compile_script.absolute()
    cmd = [str(script_path)]
    if track_changes and doc_type == "revision":
        cmd.append("--track-changes")

    logger.info(f"Running compilation: {' '.join(cmd)}")
    logger.info(f"Working directory: {project_dir}")

    try:
        import os

        cwd_original = Path.cwd()
        os.chdir(project_dir)

        try:
            result_dict = sh(
                cmd,
                verbose=True,
                return_as="dict",
                timeout=timeout * 1000,
            )

            result = type(
                "Result",
                (),
                {
                    "returncode": result_dict["exit_code"],
                    "stdout": result_dict["stdout"],
                    "stderr": result_dict["stderr"],
                },
            )()

            duration = (datetime.now() - start_time).total_seconds()
        finally:
            os.chdir(cwd_original)

        # Determine output paths
        doc_dir = project_dir / DOC_TYPE_DIRS[doc_type]

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

        # Parse errors and warnings using parse_latex_logs module
        error_issues, warning_issues = parse_compilation_output(
            result.stdout + result.stderr, log_file=log_file
        )

        # Convert LaTeXIssue objects to strings for backward compatibility
        errors = [str(issue) for issue in error_issues]
        warnings = [str(issue) for issue in warning_issues]

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
            warnings=warnings,
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

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Compilation error: {e}")
        return CompilationResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr=str(e),
            duration=duration,
        )


def compile_manuscript(project_dir: Path, timeout: int = 300) -> CompilationResult:
    """
    Compile manuscript document.

    Args:
        project_dir: Path to writer project directory (containing 01_manuscript/)
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
    return _run_compile("manuscript", project_dir, timeout=timeout)


def compile_supplementary(project_dir: Path, timeout: int = 300) -> CompilationResult:
    """
    Compile supplementary materials.

    Args:
        project_dir: Path to writer project directory (containing 02_supplementary/)
        timeout: Timeout in seconds

    Returns:
        CompilationResult with compilation status and outputs
    """
    return _run_compile("supplementary", project_dir, timeout=timeout)


def compile_revision(
    project_dir: Path, track_changes: bool = False, timeout: int = 300
) -> CompilationResult:
    """
    Compile revision responses.

    Args:
        project_dir: Path to writer project directory (containing 03_revision/)
        track_changes: Whether to enable change tracking
        timeout: Timeout in seconds

    Returns:
        CompilationResult with compilation status and outputs
    """
    return _run_compile(
        "revision", project_dir, timeout=timeout, track_changes=track_changes
    )


def run_session() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, rng
    import sys
    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


def main(args):
    project_dir = Path(args.dir) if args.dir else Path.cwd()

    if args.document == "manuscript":
        result = compile_manuscript(project_dir, timeout=args.timeout)
    elif args.document == "supplementary":
        result = compile_supplementary(project_dir, timeout=args.timeout)
    elif args.document == "revision":
        result = compile_revision(
            project_dir, track_changes=args.track_changes, timeout=args.timeout
        )

    if result.success:
        print(f"Compilation successful: {result.output_pdf}")
        return 0
    else:
        print(f"Compilation failed (exit code {result.exit_code})")
        if result.errors:
            print(f"Errors: {len(result.errors)}")
        return result.exit_code


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile LaTeX documents for scitex writer project"
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=None,
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--document",
        "-t",
        type=str,
        choices=["manuscript", "supplementary", "revision"],
        default="manuscript",
        help="Document type to compile (default: manuscript)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Compilation timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--track-changes",
        action="store_true",
        help="Enable change tracking for revision (revision only)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_session()


__all__ = [
    "CompilationResult",
    "compile_manuscript",
    "compile_supplementary",
    "compile_revision",
]

# python -m scitex.writer._compile --dir ./my_paper --document manuscript

# EOF
