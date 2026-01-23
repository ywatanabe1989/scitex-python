#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_class_hierarchy.py

"""Class hierarchy analysis utilities."""

from __future__ import annotations

import inspect

from ._resolve import get_type_info, resolve_object


def get_class_hierarchy(
    dotted_path: str,
    include_builtins: bool = False,
    max_depth: int = 10,
) -> dict:
    """
    Get the inheritance hierarchy of a class.

    Shows both parent classes (MRO) and known subclasses.

    Parameters
    ----------
    dotted_path : str
        Dotted path to the class (e.g., 'pandas.DataFrame')
    include_builtins : bool
        Include builtin classes like object, type in hierarchy
    max_depth : int
        Maximum depth for subclass traversal

    Returns
    -------
    dict
        mro: list[str] - Method Resolution Order (parent classes)
        subclasses: list[dict] - Known subclasses (recursive)
        type_info: dict

    Examples
    --------
    >>> get_class_hierarchy("collections.abc.Mapping")
    """
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    type_info = get_type_info(obj)

    if not inspect.isclass(obj):
        return {
            "success": False,
            "error": f"'{dotted_path}' is not a class",
            "type_info": type_info,
        }

    # Get MRO (parent classes)
    mro = []
    for cls in inspect.getmro(obj):
        if not include_builtins and cls.__module__ == "builtins":
            continue
        mro.append(
            {
                "name": cls.__name__,
                "module": cls.__module__,
                "qualname": f"{cls.__module__}.{cls.__name__}",
            }
        )

    # Get subclasses recursively
    subclasses = _get_subclasses_recursive(obj, max_depth, include_builtins)

    return {
        "success": True,
        "class": dotted_path,
        "mro": mro,
        "mro_count": len(mro),
        "subclasses": subclasses,
        "subclass_count": _count_subclasses(subclasses),
        "type_info": type_info,
    }


def _get_subclasses_recursive(
    cls: type,
    max_depth: int,
    include_builtins: bool,
    current_depth: int = 0,
) -> list[dict]:
    """Recursively get all subclasses."""
    if current_depth >= max_depth:
        return []

    result = []
    try:
        for sub in cls.__subclasses__():
            if not include_builtins and sub.__module__ == "builtins":
                continue

            sub_info = {
                "name": sub.__name__,
                "module": sub.__module__,
                "qualname": f"{sub.__module__}.{sub.__name__}",
            }

            children = _get_subclasses_recursive(
                sub, max_depth, include_builtins, current_depth + 1
            )
            if children:
                sub_info["subclasses"] = children

            result.append(sub_info)
    except Exception:
        pass

    return result


def _count_subclasses(subclasses: list[dict]) -> int:
    """Count total subclasses including nested."""
    count = len(subclasses)
    for sub in subclasses:
        if "subclasses" in sub:
            count += _count_subclasses(sub["subclasses"])
    return count


def get_mro(dotted_path: str, include_builtins: bool = False) -> dict:
    """
    Get just the Method Resolution Order (parent classes).

    Simpler version of get_class_hierarchy for just parents.

    Parameters
    ----------
    dotted_path : str
        Dotted path to the class
    include_builtins : bool
        Include builtin classes

    Returns
    -------
    dict
        mro: list[str] - Qualified names in MRO order
    """
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    if not inspect.isclass(obj):
        return {"success": False, "error": f"'{dotted_path}' is not a class"}

    mro = []
    for cls in inspect.getmro(obj):
        if not include_builtins and cls.__module__ == "builtins":
            continue
        mro.append(f"{cls.__module__}.{cls.__name__}")

    return {
        "success": True,
        "class": dotted_path,
        "mro": mro,
    }
