#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-11 15:15:00
# File: /home/ywatanabe/proj/scitex-code/src/scitex/msword/__init__.py

"""
MS Word (DOCX) import/export utilities for SciTeX.

This module provides high-level functions to convert between
MS Word .docx files and SciTeX's internal writer document model.

Strategy:
---------
- Word users write text only (paragraphs, minimal formatting)
- SciTeX handles: figures, tables, references, LaTeX generation
- SciTeX JSON is the "source of truth", Word is just a view/edit layer

Typical usage:
--------------
    from scitex.msword import load_docx, save_docx, list_profiles

    # Import from Word
    doc = load_docx("input.docx", profile="generic")

    # Manipulate via scitex.writer...
    # doc.normalize()

    # Export to Word (different journal template)
    save_docx(doc, "output.docx", profile="mdpi-ijerph")

Available profiles:
-------------------
- generic: Standard Word with Heading 1/2/3
- mdpi-ijerph: MDPI IJERPH journal template
- resna-2025: RESNA 2025 scientific paper template
- iop-double-anonymous: IOP double-anonymous template
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .profiles import (
    BaseWordProfile,
    get_profile,
    list_profiles,
    register_profile,
)
from .reader import WordReader
from .writer import WordWriter
from .utils import (
    link_captions_to_images,
    link_captions_to_images_by_proximity,
    normalize_section_headings,
    validate_document,
    create_post_import_hook,
)


def load_docx(
    path: str | Path,
    profile: str | None = None,
    extract_images: bool = True,
) -> dict[str, Any]:
    """
    Load a DOCX file and convert it into a SciTeX writer document.

    Parameters
    ----------
    path : str | Path
        Path to the .docx file.
    profile : str | None
        Optional profile name that specifies how to interpret Word styles
        (e.g., "mdpi-ijerph", "resna-2025"). If None, "generic" is used.
    extract_images : bool
        If True, extract embedded images and store references.

    Returns
    -------
    dict
        A SciTeX writer document structure containing:
        - blocks: List of document blocks (headings, paragraphs, captions, etc.)
        - metadata: Profile and source file information
        - images: Extracted image references (if extract_images=True)
        - references: Parsed reference entries

    Examples
    --------
    >>> from scitex.msword import load_docx
    >>> doc = load_docx("manuscript.docx", profile="mdpi-ijerph")
    >>> print(doc["metadata"]["profile"])
    'mdpi-ijerph'
    """
    path = Path(path)
    profile_obj: BaseWordProfile = get_profile(profile)
    reader = WordReader(profile=profile_obj, extract_images=extract_images)
    return reader.read(path)


def save_docx(
    writer_doc: dict[str, Any] | Any,
    path: str | Path,
    profile: str | None = None,
    overwrite: bool = True,
    template_path: str | Path | None = None,
) -> Path:
    """
    Save a SciTeX writer document as a DOCX file.

    Parameters
    ----------
    writer_doc : dict | Any
        SciTeX writer document instance to export.
    path : str | Path
        Output path for the .docx file.
    profile : str | None
        Optional profile name that controls how sections, headings,
        figures, tables and references are mapped to Word styles.
        If None, "generic" is used.
    overwrite : bool
        If False and the file already exists, raises FileExistsError.
    template_path : str | Path | None
        Optional path to a Word template (.dotx/.docx) to use as base.
        This allows using journal-specific formatting.

    Returns
    -------
    Path
        The path to the written .docx file.

    Examples
    --------
    >>> from scitex.msword import save_docx
    >>> save_docx(doc, "submission_resna_2025.docx", profile="resna-2025")
    PosixPath('submission_resna_2025.docx')
    """
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")

    profile_obj: BaseWordProfile = get_profile(profile)
    writer = WordWriter(profile=profile_obj, template_path=template_path)
    writer.write(writer_doc, output_path)
    return output_path


def convert_docx_to_tex(
    input_path: str | Path,
    output_path: str | Path,
    profile: str | None = None,
    *,
    image_dir: str | Path | None = None,
    link_images: bool = True,
    link_mode: str = "by-number",
    normalize_headings: bool = True,
    validate: bool = True,
) -> Path:
    """
    Convert a DOCX file directly to LaTeX.

    This is a convenience function that:
    1. Loads the DOCX file into SciTeX intermediate format
    2. (Optionally) normalizes headings
    3. (Optionally) links figure captions to images
    4. (Optionally) validates the document and adds warnings
    5. Exports to LaTeX (including figures via image_dir)

    Parameters
    ----------
    input_path : str | Path
        Path to the input .docx file.
    output_path : str | Path
        Path for the output .tex file.
    profile : str | None
        Word profile for interpreting styles
        (e.g., "resna-2025", "iop-double-anonymous").
    image_dir : str | Path | None, optional
        Directory where extracted figure image files will be saved.
        If None, the LaTeX exporter will create "<tex_stem>_figures"
        next to `output_path`.
    link_images : bool, default True
        Whether to link figure captions to extracted images so that
        LaTeX can generate \\includegraphics inside figure environments.
    link_mode : {"by-number", "by-proximity"}, default "by-number"
        Strategy for linking captions to images:
        - "by-number": Figure 1 -> first image, Figure 2 -> second image...
        - "by-proximity": assign images in document order, useful when
          figure numbers and image order don't match.
    normalize_headings : bool, default True
        If True, apply common heading normalizations
        (e.g., "intro" -> "Introduction").
    validate : bool, default True
        If True, run basic structural checks and populate
        doc["warnings"] with any issues.

    Returns
    -------
    Path
        The path to the written .tex file.

    Examples
    --------
    >>> from scitex.msword import convert_docx_to_tex
    >>> convert_docx_to_tex(
    ...     "RESNA 2025 Scientific Paper Template.docx",
    ...     "manuscript.tex",
    ...     profile="resna-2025",
    ...     image_dir="figures",
    ... )
    PosixPath('manuscript.tex')
    """
    # Import here to avoid circular imports
    from scitex.tex import export_tex

    # 1. DOCX -> SciTeX intermediate format
    doc = load_docx(input_path, profile=profile, extract_images=True)

    # 2. Normalize headings (optional)
    if normalize_headings:
        doc = normalize_section_headings(doc)

    # 3. Link captions to images (optional)
    if link_images and doc.get("images"):
        if link_mode == "by-proximity":
            doc = link_captions_to_images_by_proximity(doc)
        else:
            # Default: link by figure number
            doc = link_captions_to_images(doc)

    # 4. Validate document structure (optional)
    if validate:
        doc = validate_document(doc)

    # 5. SciTeX -> LaTeX (with figures)
    return export_tex(doc, output_path, image_dir=image_dir)


__all__ = [
    "load_docx",
    "save_docx",
    "convert_docx_to_tex",
    "list_profiles",
    "get_profile",
    "register_profile",
    "BaseWordProfile",
    "WordReader",
    "WordWriter",
    # Utility functions for post-processing
    "link_captions_to_images",
    "link_captions_to_images_by_proximity",
    "normalize_section_headings",
    "validate_document",
    "create_post_import_hook",
]
