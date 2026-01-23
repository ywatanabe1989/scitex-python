#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_resolve.py

"""Object resolution and type information utilities."""

from __future__ import annotations

import importlib
import inspect
from typing import Any


def resolve_object(dotted_path: str) -> tuple[Any, str | None]:
    """
    Resolve a dotted path to a Python object.

    Parameters
    ----------
    dotted_path : str
        Dotted path like 'scitex.plt.plot' or 'scitex.audio'

    Returns
    -------
    tuple[Any, str | None]
        (resolved_object, error_message)
        If successful, error_message is None

    Examples
    --------
    >>> obj, err = resolve_object("scitex.plt")
    >>> obj, err = resolve_object("scitex.audio.speak")
    """
    parts = dotted_path.split(".")
    obj = None
    last_error = None

    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_path)
            for attr_name in parts[i:]:
                obj = getattr(obj, attr_name)
            return obj, None
        except (ImportError, AttributeError) as e:
            last_error = str(e)
            continue

    return None, f"Could not resolve '{dotted_path}': {last_error}"


def get_type_info(obj: Any) -> dict:
    """
    Get type information about an object.

    Returns
    -------
    dict
        type: str - The type name
        kind: str - 'module', 'class', 'function', 'method', 'property', 'data'
        module: str - Module where defined
        qualname: str - Qualified name
    """
    type_name = type(obj).__name__

    if inspect.ismodule(obj):
        kind = "module"
    elif inspect.isclass(obj):
        kind = "class"
    elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
        kind = "function"
    elif inspect.ismethod(obj):
        kind = "method"
    elif isinstance(obj, property):
        kind = "property"
    elif callable(obj):
        kind = "callable"
    else:
        kind = "data"

    module_name = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))

    return {
        "type": type_name,
        "kind": kind,
        "module": module_name,
        "qualname": qualname,
    }
