#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_runner.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = "./src/scitex/writer/_compile/_runner.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Compilation script execution.

Executes LaTeX compilation scripts and captures results.
"""

from pathlib import Path
from datetime import datetime

from scitex.logging import getLogger
from scitex._sh import sh
from scitex.writer.dataclasses.config import DOC_TYPE_DIRS
from scitex.writer.dataclasses import CompilationResult
from ._validator import validate_before_compile
from ._parser import parse_output

logger = getLogger(__name__)


def _get_compile_script(project_dir: Path, doc_type: str) -> Path:
    """
    Get compile script path for document type.

    Parameters
    ----------
    project_dir : Path
        Path to project directory
    doc_type : str
        Document type ('manuscript', 'supplementary', 'revision')

    Returns
    -------
    Path
        Path to compilation script
    """
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
    return script_map.get(doc_type)


def _find_output_files(
    project_dir: Path,
    doc_type: str,
) -> tuple:
    """
    Find generated output files after compilation.

    Parameters
    ----------
    project_dir : Path
        Path to project directory
    doc_type : str
        Document type

    Returns
    -------
    tuple
        (output_pdf, diff_pdf, log_file)
    """
    doc_dir = project_dir / DOC_TYPE_DIRS[doc_type]

    # Find generated PDF
    pdf_name = f"{doc_type}.pdf"
    potential_pdf = doc_dir / pdf_name
    output_pdf = potential_pdf if potential_pdf.exists() else None

    # Check for diff PDF
    diff_name = f"{doc_type}_diff.pdf"
    potential_diff = doc_dir / diff_name
    diff_pdf = potential_diff if potential_diff.exists() else None

    # Find log file
    log_dir = doc_dir / "logs"
    log_file = None
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            log_file = max(log_files, key=lambda p: p.stat().st_mtime)

    return output_pdf, diff_pdf, log_file


def run_compile(
    doc_type: str,
    project_dir: Path,
    timeout: int = 300,
    track_changes: bool = False,
) -> CompilationResult:
    """
    Run compilation script and parse results.

    Parameters
    ----------
    doc_type : str
        Document type ('manuscript', 'supplementary', 'revision')
    project_dir : Path
        Path to project directory (containing 01_manuscript/, etc.)
    timeout : int
        Timeout in seconds
    track_changes : bool
        Enable change tracking (revision only)

    Returns
    -------
    CompilationResult
        Compilation status and outputs
    """
    start_time = datetime.now()
    project_dir = Path(project_dir).absolute()

    # Validate project structure before compilation
    try:
        validate_before_compile(project_dir)
    except Exception as e:
        return CompilationResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr=str(e),
            duration=0.0,
        )

    # Get compile script
    compile_script = _get_compile_script(project_dir, doc_type)
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

    # Build command
    script_path = compile_script.absolute()
    cmd = [str(script_path)]
    if track_changes and doc_type == "revision":
        cmd.append("--track-changes")

    logger.info(f"Running compilation: {' '.join(cmd)}")
    logger.info(f"Working directory: {project_dir}")

    try:
        cwd_original = Path.cwd()
        os.chdir(project_dir)

        try:
            result_dict = sh(
                cmd,
                verbose=True,
                return_as="dict",
                timeout=timeout * 1000,
            )

            result = type('Result', (), {
                'returncode': result_dict['exit_code'],
                'stdout': result_dict['stdout'],
                'stderr': result_dict['stderr']
            })()

            duration = (datetime.now() - start_time).total_seconds()
        finally:
            os.chdir(cwd_original)

        # Find output files
        output_pdf, diff_pdf, log_file = _find_output_files(
            project_dir, doc_type
        ) if result.returncode == 0 else (None, None, None)

        # Parse errors and warnings
        errors, warnings = parse_output(
            result.stdout, result.stderr, log_file=log_file
        )

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
            logger.error(
                f"Compilation failed with exit code {result.returncode}"
            )
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


__all__ = ["run_compile"]

# EOF
