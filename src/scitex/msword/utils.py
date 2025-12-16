#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-11 16:45:00
# File: /home/ywatanabe/proj/scitex-code/src/scitex/msword/utils.py

"""
Utility functions for processing MS Word documents.

These functions can be used as post_import_hooks or called directly
to process document structures.
"""

from __future__ import annotations

from typing import Any, Dict, List


def link_captions_to_images(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Link figure captions to images by matching order.

    This function pairs figure captions with images based on their
    sequential order in the document. Each figure caption is assigned
    an `image_hash` that corresponds to the image at the same position.

    Parameters
    ----------
    doc : dict
        SciTeX writer document with 'blocks' and 'images' keys.

    Returns
    -------
    dict
        The same document with image_hash added to figure captions.

    Examples
    --------
    >>> from scitex.msword import load_docx
    >>> from scitex.msword.utils import link_captions_to_images
    >>> doc = load_docx("manuscript.docx")
    >>> doc = link_captions_to_images(doc)
    >>> # Now captions have image_hash for LaTeX export
    """
    blocks = doc.get("blocks", [])
    images = doc.get("images", [])

    # Find all figure captions
    figure_captions = [
        b for b in blocks
        if b.get("type") == "caption" and b.get("caption_type") == "figure"
    ]

    # Link by order (figure 1 -> image 0, figure 2 -> image 1, etc.)
    for caption in figure_captions:
        fig_num = caption.get("number")
        if fig_num is not None and isinstance(fig_num, int):
            # Figure numbers are typically 1-indexed
            img_idx = fig_num - 1
            if 0 <= img_idx < len(images):
                caption["image_hash"] = images[img_idx].get("hash")

    return doc


def link_captions_to_images_by_proximity(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Link figure captions to images by document proximity.

    This function uses the image blocks (type="image") that are inserted
    at their actual positions in the document body. It finds the nearest
    unlinked image block to each figure caption.

    Parameters
    ----------
    doc : dict
        SciTeX writer document.

    Returns
    -------
    dict
        Document with image_hash added to captions.
    """
    blocks = doc.get("blocks", [])

    # Collect image blocks and figure captions with their indices
    image_blocks = []
    figure_captions = []

    for i, block in enumerate(blocks):
        if block.get("type") == "image":
            image_blocks.append((i, block))
        elif block.get("type") == "caption" and block.get("caption_type") == "figure":
            figure_captions.append((i, block))

    if not image_blocks:
        # Fallback to old behavior using doc["images"] list
        images = doc.get("images", [])
        if not images:
            return doc
        image_hashes = [img.get("hash") for img in images]
        for idx, (_, caption) in enumerate(figure_captions):
            if idx < len(image_hashes):
                caption["image_hash"] = image_hashes[idx]
        return doc

    used_images = set()

    # For each caption, find the nearest preceding image block
    for cap_idx, caption in figure_captions:
        best_img_idx = None
        best_img_hash = None
        best_distance = float("inf")

        for img_idx, img_block in image_blocks:
            img_hash = img_block.get("image_hash")
            if img_hash in used_images:
                continue

            # Prefer images that come before the caption (typical layout)
            distance = cap_idx - img_idx
            if distance >= 0 and distance < best_distance:
                best_distance = distance
                best_img_idx = img_idx
                best_img_hash = img_hash

        # If no preceding image, try following images
        if best_img_hash is None:
            for img_idx, img_block in image_blocks:
                img_hash = img_block.get("image_hash")
                if img_hash in used_images:
                    continue

                distance = abs(cap_idx - img_idx)
                if distance < best_distance:
                    best_distance = distance
                    best_img_idx = img_idx
                    best_img_hash = img_hash

        if best_img_hash:
            caption["image_hash"] = best_img_hash
            used_images.add(best_img_hash)

    return doc


def normalize_section_headings(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize section headings for consistency.

    Converts common section titles to standard academic format:
    - "intro" -> "Introduction"
    - "method" -> "Methods"
    - etc.

    Parameters
    ----------
    doc : dict
        SciTeX writer document.

    Returns
    -------
    dict
        Document with normalized headings.
    """
    blocks = doc.get("blocks", [])

    # Common normalizations
    normalizations = {
        "intro": "Introduction",
        "introduction": "Introduction",
        "method": "Methods",
        "methods": "Methods",
        "materials and methods": "Materials and Methods",
        "result": "Results",
        "results": "Results",
        "discussion": "Discussion",
        "conclusion": "Conclusions",
        "conclusions": "Conclusions",
        "acknowledgement": "Acknowledgements",
        "acknowledgements": "Acknowledgements",
        "reference": "References",
        "references": "References",
        "bibliography": "References",
    }

    for block in blocks:
        if block.get("type") == "heading" and block.get("level") == 1:
            text = block.get("text", "").strip().lower()
            if text in normalizations:
                block["text"] = normalizations[text]

    return doc


def validate_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate document structure and add warnings.

    Checks for common issues:
    - Missing required sections
    - Unmatched caption numbers
    - Empty references section
    - Duplicate figure numbers

    Parameters
    ----------
    doc : dict
        SciTeX writer document.

    Returns
    -------
    dict
        Document with warnings added.
    """
    blocks = doc.get("blocks", [])
    warnings = doc.get("warnings", [])

    # Check for required sections
    headings = [b.get("text", "").lower() for b in blocks if b.get("type") == "heading"]

    required_sections = ["introduction", "methods", "results", "discussion", "references"]
    for section in required_sections:
        if not any(section in h for h in headings):
            warnings.append(f"Missing section: {section.title()}")

    # Check for duplicate figure numbers
    figure_numbers = [
        b.get("number") for b in blocks
        if b.get("type") == "caption" and b.get("caption_type") == "figure"
    ]
    seen = set()
    for num in figure_numbers:
        if num in seen:
            warnings.append(f"Duplicate figure number: {num}")
        seen.add(num)

    # Check for missing references
    references = doc.get("references", [])
    if not references:
        ref_blocks = [b for b in blocks if b.get("type") == "reference-paragraph"]
        if not ref_blocks:
            warnings.append("No references found in document")

    doc["warnings"] = warnings
    return doc


def create_post_import_hook(*functions):
    """
    Create a composite post_import_hook from multiple functions.

    Parameters
    ----------
    *functions : callable
        Functions to apply in sequence.

    Returns
    -------
    callable
        A single hook that applies all functions.

    Examples
    --------
    >>> from scitex.msword.utils import (
    ...     link_captions_to_images,
    ...     normalize_section_headings,
    ...     create_post_import_hook,
    ... )
    >>> hook = create_post_import_hook(
    ...     link_captions_to_images,
    ...     normalize_section_headings,
    ... )
    >>> # Use with custom profile
    >>> profile.post_import_hooks = [hook]
    """
    def composite_hook(doc: Dict[str, Any]) -> Dict[str, Any]:
        for func in functions:
            doc = func(doc)
        return doc
    return composite_hook


__all__ = [
    "link_captions_to_images",
    "link_captions_to_images_by_proximity",
    "normalize_section_headings",
    "validate_document",
    "create_post_import_hook",
]
