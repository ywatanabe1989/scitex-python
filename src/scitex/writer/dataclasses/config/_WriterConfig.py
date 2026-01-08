#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:47 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_WriterConfig.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/_WriterConfig.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
WriterConfig - dataclass for writer configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class WriterConfig:
    """Configuration for scitex.writer."""

    project_dir: Path
    """Root directory of writer project"""

    manuscript_dir: Path
    """Directory for manuscript (01_manuscript/)"""

    supplementary_dir: Path
    """Directory for supplementary (02_supplementary/)"""

    revision_dir: Path
    """Directory for revision (03_revision/)"""

    shared_dir: Path
    """Directory for shared resources (00_shared/)"""

    compile_script: Optional[Path] = None
    """Path to compile script (auto-detected if None)"""

    @classmethod
    def from_directory(cls, project_dir: Path) -> "WriterConfig":
        """
        Create config from project directory.

        Args:
            project_dir: Path to writer project root

        Returns:
            WriterConfig instance

        Examples:
            >>> config = WriterConfig.from_directory(Path("/path/to/project"))
            >>> print(config.manuscript_dir)
        """
        project_dir = Path(project_dir)

        return cls(
            project_dir=project_dir,
            manuscript_dir=project_dir / "01_manuscript",
            supplementary_dir=project_dir / "02_supplementary",
            revision_dir=project_dir / "03_revision",
            shared_dir=project_dir / "00_shared",
        )

    def validate(self) -> bool:
        """
        Validate that required directories exist.

        Returns:
            True if valid writer project structure

        Raises:
            ValueError: If invalid structure
        """
        if not self.project_dir.exists():
            raise ValueError(f"Project directory not found: {self.project_dir}")

        # Check for at least one document directory
        doc_dirs = [
            self.manuscript_dir,
            self.supplementary_dir,
            self.revision_dir,
        ]

        if not any(d.exists() for d in doc_dirs):
            raise ValueError(
                f"No document directories found in {self.project_dir}. "
                "Expected: 01_manuscript/, 02_supplementary/, or 03_revision/"
            )

        return True


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
    config = WriterConfig.from_directory(Path(args.dir))

    print(f"Project dir: {config.project_dir}")
    print(f"Manuscript dir: {config.manuscript_dir}")
    print(f"Supplementary dir: {config.supplementary_dir}")
    print(f"Revision dir: {config.revision_dir}")
    print(f"Shared dir: {config.shared_dir}")

    if args.validate:
        try:
            config.validate()
            print("\nValidation: PASSED")
        except ValueError as ee:
            print(f"\nValidation: FAILED - {ee}")
            return 1

    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Demonstrate WriterConfig dataclass")
    parser.add_argument(
        "--dir",
        type=str,
        default="./my_paper",
        help="Project directory (default: ./my_paper)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate directory structure",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_session()


__all__ = ["WriterConfig"]

# python -m scitex.writer.dataclasses.config._WriterConfig --dir ./my_paper --validate

# EOF
