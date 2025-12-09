#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 12:27:18 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/io/_save_modules/_json.py

import json
import re
from typing import Any


class PrecisionEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that formats FixedFloat objects with fixed decimal places.

    This encoder produces output like:
        "position": [0.250, 0.294, 0.500, 0.412]

    instead of:
        "position": [0.25, 0.294, 0.5, 0.412]
    """
    def default(self, obj):
        # Handle FixedFloat from _collect_figure_metadata
        if hasattr(obj, 'value') and hasattr(obj, 'precision'):
            # Return a special marker that we'll post-process
            return f"__FIXED_{obj.precision}_{obj.value}__"
        return super().default(obj)


def _format_fixed_floats(json_str: str) -> str:
    """
    Post-process JSON string to format FixedFloat markers with proper precision.

    Converts: "__FIXED_3_0.25__" -> 0.250
    """
    pattern = r'"__FIXED_(\d+)_([-\d.]+)__"'

    def replacer(match):
        precision = int(match.group(1))
        value = float(match.group(2))
        return f"{value:.{precision}f}"

    return re.sub(pattern, replacer, json_str)


def _convert_to_serializable(obj: Any) -> Any:
    """
    Convert non-JSON-serializable objects to serializable format.

    Handles:
    - FixedFloat -> marker string for post-processing
    - DotDict -> dict
    - DataFrame -> dict of lists
    - numpy arrays -> lists
    - etc.
    """
    # Handle FixedFloat (from _collect_figure_metadata)
    if hasattr(obj, 'value') and hasattr(obj, 'precision') and hasattr(obj, '__class__') and obj.__class__.__name__ == 'FixedFloat':
        return f"__FIXED_{obj.precision}_{obj.value}__"

    # Handle DotDict
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "DotDict":
        return _convert_to_serializable(dict(obj))

    # Handle pandas DataFrame
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "DataFrame":
        return obj.to_dict("list")

    # Handle numpy arrays
    if hasattr(obj, "tolist"):
        return obj.tolist()

    # Handle dict recursively
    if isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}

    # Handle list recursively
    if isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]

    # Handle tuple
    if isinstance(obj, tuple):
        return list(obj)

    # Default: return as-is and let json.dump handle or error
    return obj


def _save_json(obj, spath):
    """
    Save a Python object as a JSON file.

    Automatically converts DotDict objects and pandas DataFrames
    to JSON-serializable formats. Supports FixedFloat objects for
    consistent decimal place formatting.

    Parameters
    ----------
    obj : dict, list, DotDict, or other JSON-serializable object
        The object to serialize to JSON.
    spath : str
        Path where the JSON file will be saved.

    Returns
    -------
    None
    """
    # Convert to serializable format
    serializable_obj = _convert_to_serializable(obj)

    # Use custom encoder and post-process for fixed-precision floats
    json_str = json.dumps(serializable_obj, indent=4, cls=PrecisionEncoder)
    json_str = _format_fixed_floats(json_str)

    with open(spath, "w") as f:
        f.write(json_str)
