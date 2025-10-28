#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/validate.py

"""
Project structure validation for writer module.

Validates that writer projects have correct directory structure before compilation.
"""

from pathlib import Path
from typing import List, Tuple
from scitex.logging import getLogger

logger = getLogger(__name__)


class ProjectValidationError(Exception):
    """Raised when project structure is invalid."""
    pass


def validate_manuscript_structure(project_dir: Path) -> bool:
    """
    Validate manuscript project structure.

    Checks:
    - 01_manuscript/ exists
    - 01_manuscript/contents/ exists

    Args:
        project_dir: Path to writer project

    Returns:
        True if valid

    Raises:
        ProjectValidationError: If invalid structure
    """
    project_dir = Path(project_dir)
    required_dirs = [
        project_dir / "01_manuscript",
        project_dir / "01_manuscript" / "contents",
    ]

    for directory in required_dirs:
        if not directory.exists():
            raise ProjectValidationError(
                f"Required directory missing: {directory}"
            )
        if not directory.is_dir():
            raise ProjectValidationError(
                f"Expected directory but found file: {directory}"
            )

    logger.debug(f"Manuscript structure valid: {project_dir}")
    return True


def validate_supplementary_structure(project_dir: Path) -> bool:
    """
    Validate supplementary project structure.

    Args:
        project_dir: Path to writer project

    Returns:
        True if valid

    Raises:
        ProjectValidationError: If invalid structure
    """
    project_dir = Path(project_dir)
    required_dirs = [
        project_dir / "02_supplementary",
        project_dir / "02_supplementary" / "contents",
    ]

    for directory in required_dirs:
        if not directory.exists():
            raise ProjectValidationError(
                f"Required directory missing: {directory}"
            )
        if not directory.is_dir():
            raise ProjectValidationError(
                f"Expected directory but found file: {directory}"
            )

    logger.debug(f"Supplementary structure valid: {project_dir}")
    return True


def validate_revision_structure(project_dir: Path) -> bool:
    """
    Validate revision project structure.

    Args:
        project_dir: Path to writer project

    Returns:
        True if valid

    Raises:
        ProjectValidationError: If invalid structure
    """
    project_dir = Path(project_dir)
    required_dirs = [
        project_dir / "03_revision",
        project_dir / "03_revision" / "contents",
    ]

    for directory in required_dirs:
        if not directory.exists():
            raise ProjectValidationError(
                f"Required directory missing: {directory}"
            )
        if not directory.is_dir():
            raise ProjectValidationError(
                f"Expected directory but found file: {directory}"
            )

    logger.debug(f"Revision structure valid: {project_dir}")
    return True


def validate_all_documents(project_dir: Path) -> bool:
    """
    Validate that at least one document type is available.

    Args:
        project_dir: Path to writer project

    Returns:
        True if at least one valid document found

    Raises:
        ProjectValidationError: If no valid documents found
    """
    project_dir = Path(project_dir)
    doc_types = {
        'manuscript': '01_manuscript',
        'supplementary': '02_supplementary',
        'revision': '03_revision',
    }

    valid_docs = []
    for doc_type, doc_dir_name in doc_types.items():
        doc_dir = project_dir / doc_dir_name
        contents_dir = doc_dir / "contents"

        if doc_dir.exists() and contents_dir.exists():
            valid_docs.append(doc_type)
            logger.debug(f"Found {doc_type} document")
        else:
            logger.warning(f"{doc_type} structure incomplete: {doc_dir}")

    if not valid_docs:
        raise ProjectValidationError(
            f"No valid document directories in {project_dir}. "
            f"Expected at least one of: 01_manuscript/, 02_supplementary/, 03_revision/"
        )

    return True


def list_missing_files(
    project_dir: Path,
    doc_type: str = 'manuscript'
) -> List[str]:
    """
    List potentially missing content files.

    Expected files for manuscript type:
    - abstract.tex
    - introduction.tex
    - methods.tex
    - results.tex
    - discussion.tex

    Args:
        project_dir: Path to writer project
        doc_type: Document type to check

    Returns:
        List of expected but missing files
    """
    project_dir = Path(project_dir)
    doc_map = {
        'manuscript': '01_manuscript',
        'supplementary': '02_supplementary',
        'revision': '03_revision',
    }

    doc_dir = project_dir / doc_map[doc_type]
    contents_dir = doc_dir / "contents"

    if not contents_dir.exists():
        logger.debug(f"No contents directory: {contents_dir}")
        return []

    # Define expected files based on type
    expected = {
        'manuscript': [
            'abstract.tex',
            'introduction.tex',
            'methods.tex',
            'results.tex',
            'discussion.tex'
        ],
        'supplementary': [],  # Can have any files
        'revision': [],       # Can have any files
    }

    missing = []
    for filename in expected.get(doc_type, []):
        if not (contents_dir / filename).exists():
            missing.append(filename)
            logger.warning(f"Missing {doc_type} file: {filename}")

    return missing


__all__ = [
    'ProjectValidationError',
    'validate_manuscript_structure',
    'validate_supplementary_structure',
    'validate_revision_structure',
    'validate_all_documents',
    'list_missing_files',
]

# EOF
