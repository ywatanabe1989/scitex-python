#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_signature.py

"""Signature extraction utilities."""

from __future__ import annotations

import inspect
from typing import Any

from ._resolve import get_type_info, resolve_object


def _format_annotation(annotation: Any) -> str:
    """Format a type annotation as a string."""
    if annotation is None:
        return "None"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _build_signature_string(
    obj: Any,
    parameters: list[dict],
    return_annotation: str | None,
) -> str:
    """Build a human-readable signature string."""
    name = getattr(obj, "__name__", "?")

    param_strs = []
    for p in parameters:
        s = p["name"]
        if "annotation" in p:
            s += f": {p['annotation']}"
        if "default" in p:
            s += f" = {p['default']}"
        param_strs.append(s)

    sig = f"{name}({', '.join(param_strs)})"
    if return_annotation:
        sig += f" -> {return_annotation}"

    return sig


def q(
    dotted_path: str,
    include_defaults: bool = True,
    include_annotations: bool = True,
) -> dict:
    """
    Get the signature of a function, method, or class.

    Like IPython's `func?` (quick info).

    Parameters
    ----------
    dotted_path : str
        Dotted path to the callable (e.g., 'scitex.plt.plot')
    include_defaults : bool
        Include default values in signature
    include_annotations : bool
        Include type annotations

    Returns
    -------
    dict
        name: str
        signature: str - Human-readable signature
        parameters: list[dict] - Detailed parameter info
        return_annotation: str | None
        type_info: dict

    Examples
    --------
    >>> q("scitex.plt.plot")
    {'name': 'plot', 'signature': 'plot(spec: dict, ...) -> dict', ...}
    """
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    type_info = get_type_info(obj)

    callable_obj = obj
    if inspect.isclass(obj):
        callable_obj = obj.__init__

    try:
        sig = inspect.signature(callable_obj)
    except (ValueError, TypeError) as e:
        return {
            "success": False,
            "error": f"Cannot get signature: {e}",
            "type_info": type_info,
        }

    parameters = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        param_info = {
            "name": name,
            "kind": str(param.kind).split(".")[-1],
        }

        if include_annotations and param.annotation != inspect.Parameter.empty:
            param_info["annotation"] = _format_annotation(param.annotation)

        if include_defaults and param.default != inspect.Parameter.empty:
            param_info["default"] = repr(param.default)

        parameters.append(param_info)

    return_annotation = None
    if include_annotations and sig.return_annotation != inspect.Signature.empty:
        return_annotation = _format_annotation(sig.return_annotation)

    sig_str = _build_signature_string(obj, parameters, return_annotation)

    return {
        "success": True,
        "name": getattr(obj, "__name__", dotted_path.split(".")[-1]),
        "signature": sig_str,
        "parameters": parameters,
        "return_annotation": return_annotation,
        "type_info": type_info,
    }
