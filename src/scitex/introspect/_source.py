#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_source.py

"""Source code retrieval utilities."""

from __future__ import annotations

import inspect

from ._resolve import get_type_info, resolve_object


def qq(
    dotted_path: str,
    max_lines: int | None = None,
    include_decorators: bool = True,
) -> dict:
    """
    Get the source code of a Python object.

    Like IPython's `func??` (full source).

    Parameters
    ----------
    dotted_path : str
        Dotted path to the object
    max_lines : int | None
        Limit output to first N lines (None = no limit)
    include_decorators : bool
        Include decorator lines

    Returns
    -------
    dict
        source: str
        file: str - Source file path
        line_start: int - Starting line number
        line_count: int - Number of lines
        type_info: dict
    """
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    type_info = get_type_info(obj)

    try:
        source = inspect.getsource(obj)
        source_file = inspect.getfile(obj)
        _, line_start = inspect.getsourcelines(obj)
    except (TypeError, OSError) as e:
        return {
            "success": False,
            "error": f"Cannot get source: {e}",
            "type_info": type_info,
        }

    lines = source.split("\n")
    line_count = len(lines)

    if not include_decorators and lines:
        i = 0
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        lines = lines[i:]
        source = "\n".join(lines)

    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        source = "\n".join(lines) + f"\n... ({line_count - max_lines} more lines)"

    return {
        "success": True,
        "source": source,
        "file": source_file,
        "line_start": line_start,
        "line_count": line_count,
        "type_info": type_info,
    }
