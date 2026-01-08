#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:38 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_Document.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/_Document.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Base Document class for document type accessors.

Provides dynamic file access via attribute lookups.
"""

from pathlib import Path
from typing import Optional

from ._DocumentSection import DocumentSection


class Document:
    """
    Base document accessor.

    Provides dynamic file access by mapping attribute names to .tex files:
    - document.introduction -> introduction.tex
    - document.methods -> methods.tex
    - Custom: document.custom_section -> custom_section.tex
    """

    def __init__(self, doc_dir: Path, git_root: Optional[Path] = None):
        """
        Initialize document accessor.

        Args:
            doc_dir: Path to document directory (contains 'contents/' subdirectory)
            git_root: Path to git repository root (optional, for efficiency)
        """
        self.dir = doc_dir
        self.git_root = git_root

    def __getattr__(self, name: str) -> DocumentSection:
        """
        Get file path by name (e.g., introduction -> introduction.tex).

        Args:
            name: Section name without .tex extension

        Returns:
            DocumentSection for the requested file

        Example:
            >>> manuscript = ManuscriptDocument(Path("01_manuscript"))
            >>> manuscript.introduction.read()  # Reads contents/introduction.tex
        """
        if name.startswith("_"):
            # Avoid infinite recursion for private attributes
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )

        file_path = self.dir / "contents" / f"{name}.tex"
        return DocumentSection(file_path, git_root=self.git_root)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.dir.name})"


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
    doc = Document(Path(args.dir))

    print(f"Document: {doc}")
    print(f"Directory: {doc.dir}")

    contents_dir = doc.dir / "contents"
    if contents_dir.exists():
        tex_files = list(contents_dir.glob("*.tex"))
        print(f"\nAvailable sections ({len(tex_files)}):")
        for tex_file in sorted(tex_files):
            section_name = tex_file.stem
            print(f"  - {section_name}")

    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate Document accessor functionality"
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        required=True,
        help="Path to document directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_session()


__all__ = ["Document"]

# python -m scitex.writer.dataclasses.core._Document --dir ./01_manuscript

# EOF
