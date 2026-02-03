#!/usr/bin/env python3
# Timestamp: 2026-02-04
# File: src/scitex/scholar/local_dbs/export.py
"""Export functionality for unified local database results.

Supports multiple output formats:
- text: Human-readable formatted text
- json: JSON format with all fields
- bibtex: BibTeX bibliography format
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from .unified import UnifiedSearchResult, UnifiedWork

__all__ = [
    "save",
    "SUPPORTED_FORMATS",
]

SUPPORTED_FORMATS = ["text", "json", "bibtex"]


def save(
    data: Union[UnifiedWork, UnifiedSearchResult, List[UnifiedWork]],
    path: str,
    format: str = "json",
) -> str:
    """Save UnifiedWork(s) or UnifiedSearchResult to a file.

    Args:
        data: UnifiedWork, UnifiedSearchResult, or list of UnifiedWorks
        path: Output file path
        format: Output format ("text", "json", "bibtex")

    Returns
    -------
        Path to saved file

    Raises
    ------
        ValueError: If format is not supported

    Examples
    --------
        >>> from scitex.scholar.local_dbs import search, save
        >>> results = search("machine learning", limit=10)
        >>> save(results, "results.json")
        >>> save(results, "results.bib", format="bibtex")
        >>> save(results, "results.txt", format="text")
    """
    from .unified import UnifiedSearchResult, UnifiedWork, to_bibtex, to_json, to_text

    if format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {format}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    path = Path(path)

    # Extract works
    if isinstance(data, UnifiedWork):
        works = [data]
    elif isinstance(data, UnifiedSearchResult):
        works = data.works
    elif isinstance(data, list):
        works = data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Generate content
    if format == "text":
        content = to_text(works)
    elif format == "json":
        content = to_json(works)
    elif format == "bibtex":
        content = to_bibtex(works)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return str(path)


# EOF
