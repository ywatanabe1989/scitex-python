#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:43 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_ManuscriptContents.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/_ManuscriptContents.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
ManuscriptContents - dataclass for manuscript contents structure.

Represents the 01_manuscript/contents/ directory structure with all files.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..core import DocumentSection


@dataclass
class ManuscriptContents:
    """Contents subdirectory of manuscript (01_manuscript/contents/)."""

    root: Path
    git_root: Optional[Path] = None

    # Core sections
    abstract: DocumentSection = None
    introduction: DocumentSection = None
    methods: DocumentSection = None
    results: DocumentSection = None
    discussion: DocumentSection = None

    # Metadata
    title: DocumentSection = None
    authors: DocumentSection = None
    keywords: DocumentSection = None
    journal_name: DocumentSection = None

    # Optional sections
    graphical_abstract: DocumentSection = None
    highlights: DocumentSection = None
    data_availability: DocumentSection = None
    additional_info: DocumentSection = None
    wordcount: DocumentSection = None

    # Files/directories
    figures: Path = None
    tables: Path = None
    bibliography: DocumentSection = None
    latex_styles: Path = None

    def __post_init__(self):
        """Initialize all DocumentSection instances."""
        if self.abstract is None:
            self.abstract = DocumentSection(self.root / "abstract.tex", self.git_root)
        if self.introduction is None:
            self.introduction = DocumentSection(
                self.root / "introduction.tex", self.git_root
            )
        if self.methods is None:
            self.methods = DocumentSection(self.root / "methods.tex", self.git_root)
        if self.results is None:
            self.results = DocumentSection(self.root / "results.tex", self.git_root)
        if self.discussion is None:
            self.discussion = DocumentSection(
                self.root / "discussion.tex", self.git_root
            )
        if self.title is None:
            self.title = DocumentSection(self.root / "title.tex", self.git_root)
        if self.authors is None:
            self.authors = DocumentSection(self.root / "authors.tex", self.git_root)
        if self.keywords is None:
            self.keywords = DocumentSection(self.root / "keywords.tex", self.git_root)
        if self.journal_name is None:
            self.journal_name = DocumentSection(
                self.root / "journal_name.tex", self.git_root
            )
        if self.graphical_abstract is None:
            self.graphical_abstract = DocumentSection(
                self.root / "graphical_abstract.tex", self.git_root
            )
        if self.highlights is None:
            self.highlights = DocumentSection(
                self.root / "highlights.tex", self.git_root
            )
        if self.data_availability is None:
            self.data_availability = DocumentSection(
                self.root / "data_availability.tex", self.git_root
            )
        if self.additional_info is None:
            self.additional_info = DocumentSection(
                self.root / "additional_info.tex", self.git_root
            )
        if self.wordcount is None:
            self.wordcount = DocumentSection(self.root / "wordcount.tex", self.git_root)
        if self.figures is None:
            self.figures = self.root / "figures"
        if self.tables is None:
            self.tables = self.root / "tables"
        if self.bibliography is None:
            self.bibliography = DocumentSection(
                self.root / "bibliography.bib", self.git_root
            )
        if self.latex_styles is None:
            self.latex_styles = self.root / "latex_styles"

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify manuscript structure has required files.

        Returns:
            (is_valid, list_of_missing_files_with_paths)
        """
        required = [
            ("abstract.tex", self.abstract),
            ("introduction.tex", self.introduction),
            ("methods.tex", self.methods),
            ("results.tex", self.results),
            ("discussion.tex", self.discussion),
        ]

        missing = []
        for name, section in required:
            if not section.path.exists():
                expected_path = (
                    section.path.relative_to(self.git_root)
                    if self.git_root
                    else section.path
                )
                missing.append(f"{name} (expected at: {expected_path})")

        return len(missing) == 0, missing


def run_session() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, rng
    import sys
    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(
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
    contents_path = Path(args.dir) / "01_manuscript" / "contents"
    git_root = Path(args.dir) if args.git_root else None

    contents = ManuscriptContents(
        root=contents_path,
        git_root=git_root,
    )

    print(f"Root: {contents.root}")
    print(f"Git root: {contents.git_root}")
    print(f"\nCore sections:")
    print(f"  Abstract: {contents.abstract.path}")
    print(f"  Introduction: {contents.introduction.path}")
    print(f"  Methods: {contents.methods.path}")
    print(f"  Results: {contents.results.path}")
    print(f"  Discussion: {contents.discussion.path}")

    if args.verify:
        is_valid, missing = contents.verify_structure()
        if is_valid:
            print("\nStructure verification: PASSED")
        else:
            print(f"\nStructure verification: FAILED")
            print(f"Missing files: {', '.join(missing)}")
            return 1

    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate ManuscriptContents dataclass"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./my_paper",
        help="Project directory (default: ./my_paper)",
    )
    parser.add_argument(
        "--git-root",
        action="store_true",
        help="Use project dir as git root",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify structure has required files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_session()


__all__ = ["ManuscriptContents"]

# python -m scitex.writer.dataclasses.contents._ManuscriptContents --dir ./my_paper --verify

# EOF
