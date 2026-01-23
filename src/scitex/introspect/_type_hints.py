#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_type_hints.py

"""Type hint analysis utilities."""

from __future__ import annotations

import inspect
import typing
from typing import Any, get_type_hints

from ._resolve import get_type_info, resolve_object


def get_type_hints_detailed(
    dotted_path: str,
    include_extras: bool = True,
) -> dict:
    """
    Get detailed type hint information for a callable or class.

    Parameters
    ----------
    dotted_path : str
        Dotted path to the function, method, or class
    include_extras : bool
        Include typing extras (Annotated metadata, etc.)

    Returns
    -------
    dict
        hints: dict[str, dict] - Parameter name to type info
        return_hint: dict | None - Return type info
        type_info: dict

    Examples
    --------
    >>> get_type_hints_detailed("json.dumps")
    """
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    type_info_obj = get_type_info(obj)

    # Get the object to analyze
    target = obj
    if inspect.isclass(obj):
        target = obj.__init__

    try:
        hints = get_type_hints(target, include_extras=include_extras)
    except Exception as e:
        # Some objects don't support get_type_hints
        hints = getattr(target, "__annotations__", {})
        if not hints:
            return {
                "success": False,
                "error": f"Cannot get type hints: {e}",
                "type_info": type_info_obj,
            }

    # Analyze each hint
    analyzed_hints = {}
    return_hint = None

    for name, hint in hints.items():
        hint_info = _analyze_type(hint)
        if name == "return":
            return_hint = hint_info
        else:
            analyzed_hints[name] = hint_info

    return {
        "success": True,
        "hints": analyzed_hints,
        "return_hint": return_hint,
        "hint_count": len(analyzed_hints),
        "type_info": type_info_obj,
    }


def _analyze_type(hint: Any) -> dict:
    """Analyze a type hint and return structured info."""
    result = {
        "raw": _format_type(hint),
        "origin": None,
        "args": [],
        "is_optional": False,
        "is_union": False,
        "is_generic": False,
    }

    # Get origin (e.g., list from list[int])
    origin = typing.get_origin(hint)
    if origin is not None:
        result["origin"] = _format_type(origin)
        result["is_generic"] = True

        # Get args (e.g., int from list[int])
        args = typing.get_args(hint)
        if args:
            result["args"] = [_format_type(a) for a in args]

        # Check for Union/Optional
        if origin is typing.Union:
            result["is_union"] = True
            # Optional is Union[X, None]
            if type(None) in args:
                result["is_optional"] = True

    return result


def _format_type(t: Any) -> str:
    """Format a type as a readable string."""
    if t is type(None):
        return "None"
    if hasattr(t, "__name__"):
        return t.__name__
    if hasattr(t, "_name"):
        return t._name or str(t)
    return str(t).replace("typing.", "")


def get_class_annotations(dotted_path: str) -> dict:
    """
    Get all annotations for a class (class vars and methods).

    Parameters
    ----------
    dotted_path : str
        Dotted path to the class

    Returns
    -------
    dict
        class_vars: dict - Class variable annotations
        methods: dict - Method annotations (name -> hints)
    """
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    if not inspect.isclass(obj):
        return {"success": False, "error": f"'{dotted_path}' is not a class"}

    # Class-level annotations
    class_vars = {}
    for name, hint in getattr(obj, "__annotations__", {}).items():
        class_vars[name] = _analyze_type(hint)

    # Method annotations
    methods = {}
    for name, member in inspect.getmembers(obj):
        if inspect.isfunction(member) or inspect.ismethod(member):
            try:
                hints = get_type_hints(member)
                if hints:
                    methods[name] = {k: _analyze_type(v) for k, v in hints.items()}
            except Exception:
                pass

    return {
        "success": True,
        "class": dotted_path,
        "class_vars": class_vars,
        "methods": methods,
        "class_var_count": len(class_vars),
        "method_count": len(methods),
    }
