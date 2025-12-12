#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata/_utils.py

"""Shared utilities for metadata handling."""

import json
from typing import Any, Dict


def convert_for_json(obj: Any) -> Any:
    """
    Convert non-JSON-serializable objects to serializable format.

    Handles FixedFloat objects from figure metadata by converting
    them to regular floats.
    """
    # Handle FixedFloat (from _collect_figure_metadata)
    if (
        hasattr(obj, "value")
        and hasattr(obj, "precision")
        and hasattr(obj, "__class__")
        and obj.__class__.__name__ == "FixedFloat"
    ):
        return obj.value

    # Handle dict recursively
    if isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}

    # Handle list recursively
    if isinstance(obj, list):
        return [convert_for_json(item) for item in obj]

    # Handle tuple
    if isinstance(obj, tuple):
        return [convert_for_json(item) for item in obj]

    # Handle numpy arrays
    if hasattr(obj, "tolist"):
        return obj.tolist()

    # Default: return as-is
    return obj


def serialize_metadata(metadata: Dict[str, Any]) -> str:
    """Serialize metadata to JSON string."""
    metadata = convert_for_json(metadata)
    try:
        return json.dumps(metadata, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Metadata must be JSON serializable: {e}")


def has_metadata(image_path: str) -> bool:
    """
    Check if an image file has embedded metadata.

    Args:
        image_path: Path to the image file

    Returns:
        True if metadata exists, False otherwise

    Example:
        >>> if has_metadata('result.png'):
        ...     print(read_metadata('result.png'))
    """
    from ._read import read_metadata

    try:
        metadata = read_metadata(image_path)
        return metadata is not None
    except:
        return False


# EOF
