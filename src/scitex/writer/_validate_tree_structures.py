#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:13:05 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_validate_tree_structures.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_validate_tree_structures.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import argparse

"""Project structure validation for writer module.

Leverages dataclass verify_structure() methods for validation."""

from pathlib import Path
from scitex.logging import getLogger
from .dataclasses import ManuscriptTree
from .dataclasses import SupplementaryTree
from .dataclasses import RevisionTree
from .dataclasses import ConfigTree
from .dataclasses import ScriptsTree
from .dataclasses import SharedTree

logger = getLogger(__name__)

# Parameters
TREE_VALIDATORS = {
    "config": {"dir_name": "config", "tree_class": ConfigTree},
    "00_shared": {"dir_name": "00_shared", "tree_class": SharedTree},
    "01_manuscript": {
        "dir_name": "01_manuscript",
        "tree_class": ManuscriptTree,
    },
    "02_supplementary": {
        "dir_name": "02_supplementary",
        "tree_class": SupplementaryTree,
    },
    "03_revision": {"dir_name": "03_revision", "tree_class": RevisionTree},
    "scripts": {"dir_name": "scripts", "tree_class": ScriptsTree},
}


# Exception classes
class ProjectValidationError(Exception):
    """Raised when project structure is invalid."""

    pass


# 2. Public validation functions
def validate_tree_structures(
    project_dir: Path, func_name="validate_tree_structures"
) -> None:
    """Validates all tree structures in the project directory."""
    logger.info(
        f"{func_name}: Validating tree structures: {Path(project_dir).absolute()}..."
    )
    project_dir = Path(project_dir)
    for dir_name, (dir_path, tree_class) in TREE_VALIDATORS.items():
        validator_func_name = f"_validate_{dir_name}_structure"
        eval(validator_func_name)(project_dir)
    logger.success(
        f"{func_name}: Validated tree structures: {Path(project_dir).absolute()}"
    )
    return True


# 3. Internal validation functions
def _validate_01_manuscript_structure(project_dir: Path) -> bool:
    """Validates manuscript structure."""
    return _validate_tree_structure_base(
        project_dir, **TREE_VALIDATORS["01_manuscript"]
    )


def _validate_02_supplementary_structure(project_dir: Path) -> bool:
    """Validates supplementary structure."""
    return _validate_tree_structure_base(
        project_dir, **TREE_VALIDATORS["02_supplementary"]
    )


def _validate_03_revision_structure(project_dir: Path) -> bool:
    """Validates revision structure."""
    return _validate_tree_structure_base(project_dir, **TREE_VALIDATORS["03_revision"])


def _validate_config_structure(project_dir: Path) -> bool:
    """Validates config structure."""
    return _validate_tree_structure_base(project_dir, **TREE_VALIDATORS["config"])


def _validate_scripts_structure(project_dir: Path) -> bool:
    """Validates scripts structure."""
    return _validate_tree_structure_base(project_dir, **TREE_VALIDATORS["scripts"])


def _validate_00_shared_structure(project_dir: Path) -> bool:
    """Validates shared structure."""
    return _validate_tree_structure_base(project_dir, **TREE_VALIDATORS["00_shared"])


# 4. Helper functions
def _validate_tree_structure_base(
    project_dir: Path, dir_name: str, tree_class: type = None
) -> bool:
    """Base validation function that checks directory existence and verifies structure using tree class.

    Args:
        project_dir: Root project directory
        dir_name: Name of directory to validate
        tree_class: Tree class with verify_structure method

    Returns:
        True if structure is valid

    Raises:
        ProjectValidationError: If directory missing or structure invalid
    """
    project_dir = Path(project_dir)
    target_dir = project_dir / dir_name
    if not target_dir.exists():
        raise ProjectValidationError(f"Required directory missing: {target_dir}")
    if tree_class is not None:
        doc = tree_class(target_dir, git_root=project_dir)
        is_valid, issues = doc.verify_structure()
        if not is_valid:
            raise ProjectValidationError(
                f"{dir_name} structure invalid:\n"
                + "\n".join(f"  - {issue}" for issue in issues)
            )
    logger.debug(f"{dir_name} structure valid: {project_dir}")
    return True


# 1. Main entry point
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
    validate_tree_structures(project_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate scitex writer project structure"
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=None,
        help="Project directory to validate (default: current directory)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run_session()

# python -m scitex.writer._validate_tree_structures --dir ./my_paper

# EOF
