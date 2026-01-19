#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_docstring.py

"""Docstring extraction and parsing utilities."""

from __future__ import annotations

import inspect
import re
from typing import Literal

from ._resolve import get_type_info, resolve_object


def _parse_docstring(docstring: str) -> dict:
    """Parse numpy/google style docstring into sections."""
    sections = {
        "summary": "",
        "description": "",
        "parameters": "",
        "returns": "",
        "examples": "",
        "notes": "",
    }

    if not docstring:
        return sections

    section_patterns = [
        (r"Parameters?\s*\n[-=]+", "parameters"),
        (r"Returns?\s*\n[-=]+", "returns"),
        (r"Examples?\s*\n[-=]+", "examples"),
        (r"Notes?\s*\n[-=]+", "notes"),
        (r"Raises?\s*\n[-=]+", "raises"),
        (r"See Also\s*\n[-=]+", "see_also"),
    ]

    lines = docstring.split("\n")

    i = 0
    summary_lines = []
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            if summary_lines:
                break
        elif any(re.match(p, line, re.IGNORECASE) for p, _ in section_patterns):
            break
        else:
            summary_lines.append(line)
        i += 1

    sections["summary"] = " ".join(summary_lines)

    current_section = "description"
    current_content = []

    for line in lines[i:]:
        matched = False
        for pattern, section_name in section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = section_name
                current_content = []
                matched = True
                break

        if not matched:
            current_content.append(line)

    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def get_docstring(
    dotted_path: str,
    format: Literal["raw", "parsed", "summary"] = "raw",
) -> dict:
    """
    Get the docstring of a Python object.

    Parameters
    ----------
    dotted_path : str
        Dotted path to the object
    format : str
        'raw' - Return full docstring as-is
        'parsed' - Parse numpy/google style into sections
        'summary' - Return only first line/paragraph

    Returns
    -------
    dict
        docstring: str
        sections: dict (if format='parsed')
        type_info: dict
    """
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    type_info = get_type_info(obj)
    docstring = inspect.getdoc(obj) or ""

    if format == "summary":
        lines = docstring.split("\n\n")
        summary = lines[0].strip() if lines else ""
        return {
            "success": True,
            "docstring": summary,
            "type_info": type_info,
        }

    if format == "parsed":
        sections = _parse_docstring(docstring)
        return {
            "success": True,
            "docstring": docstring,
            "sections": sections,
            "type_info": type_info,
        }

    return {
        "success": True,
        "docstring": docstring,
        "type_info": type_info,
    }
