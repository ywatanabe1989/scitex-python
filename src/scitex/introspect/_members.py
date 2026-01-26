#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_members.py

"""Member listing utilities."""

from __future__ import annotations

import builtins
import inspect
from typing import Literal

from ._resolve import get_type_info, resolve_object

# Save reference to built-in dir before shadowing
_builtin_dir = builtins.dir


def dir(
    dotted_path: str,
    filter: Literal["all", "public", "private", "dunder"] = "public",
    kind: Literal["all", "functions", "classes", "data", "modules"] | None = None,
    include_inherited: bool = False,
) -> dict:
    """
    List members of a module or class.

    Like Python's `dir()` but with filtering and metadata.

    Parameters
    ----------
    dotted_path : str
        Dotted path to the module or class
    filter : str
        'all' - All members
        'public' - Only public (no leading _)
        'private' - Only private (single _)
        'dunder' - Only dunder (__name__)
    kind : str | None
        Filter by type: 'functions', 'classes', 'data', 'modules'
    include_inherited : bool
        For classes, include inherited members

    Returns
    -------
    dict
        members: list[dict] - Each with name, kind, summary
        count: int
        type_info: dict
    """
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    type_info = get_type_info(obj)

    if inspect.isclass(obj) and not include_inherited:
        member_names = list(obj.__dict__.keys())
    else:
        member_names = _builtin_dir(obj)

    if filter == "public":
        member_names = [n for n in member_names if not n.startswith("_")]
    elif filter == "private":
        member_names = [
            n for n in member_names if n.startswith("_") and not n.startswith("__")
        ]
    elif filter == "dunder":
        member_names = [
            n for n in member_names if n.startswith("__") and n.endswith("__")
        ]

    members = []
    for name in sorted(member_names):
        try:
            member = getattr(obj, name)
        except AttributeError:
            continue

        member_type_info = get_type_info(member)
        member_kind = member_type_info["kind"]

        if kind:
            kind_map = {
                "functions": ("function", "method", "builtin_function_or_method"),
                "classes": ("class",),
                "data": ("data",),
                "modules": ("module",),
            }
            if kind in kind_map and member_kind not in kind_map[kind]:
                continue

        doc = inspect.getdoc(member) or ""
        summary = doc.split("\n")[0] if doc else ""

        members.append(
            {
                "name": name,
                "kind": member_kind,
                "summary": summary[:100] + "..." if len(summary) > 100 else summary,
            }
        )

    return {
        "success": True,
        "members": members,
        "count": len(members),
        "type_info": type_info,
    }


def get_exports(dotted_path: str) -> dict:
    """
    Get the __all__ exports of a module.

    Parameters
    ----------
    dotted_path : str
        Dotted path to the module

    Returns
    -------
    dict
        exports: list[str] - Names in __all__
        has_all: bool - Whether __all__ is defined
        type_info: dict
    """
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    type_info = get_type_info(obj)

    if not inspect.ismodule(obj):
        return {
            "success": False,
            "error": f"'{dotted_path}' is not a module",
            "type_info": type_info,
        }

    exports = getattr(obj, "__all__", None)

    if exports is None:
        exports = [n for n in _builtin_dir(obj) if not n.startswith("_")]
        has_all = False
    else:
        has_all = True

    return {
        "success": True,
        "exports": list(exports),
        "has_all": has_all,
        "count": len(exports),
        "type_info": type_info,
    }
