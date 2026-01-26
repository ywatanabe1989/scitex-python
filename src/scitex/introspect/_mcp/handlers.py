#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_mcp/handlers.py

"""MCP handlers for introspection tools."""

from __future__ import annotations

from typing import Literal


async def q_handler(
    dotted_path: str,
    include_defaults: bool = True,
    include_annotations: bool = True,
) -> dict:
    """Get the signature of a function, method, or class (like IPython's func?)."""
    try:
        from .. import q

        result = q(
            dotted_path,
            include_defaults=include_defaults,
            include_annotations=include_annotations,
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def docstring_handler(
    dotted_path: str,
    format: Literal["raw", "parsed", "summary"] = "raw",
) -> dict:
    """Get the docstring of a Python object."""
    try:
        from .. import get_docstring

        result = get_docstring(dotted_path, format=format)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def qq_handler(
    dotted_path: str,
    max_lines: int | None = None,
    include_decorators: bool = True,
) -> dict:
    """Get the source code of a Python object (like IPython's func??)."""
    try:
        from .. import qq

        result = qq(
            dotted_path,
            max_lines=max_lines,
            include_decorators=include_decorators,
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def dir_handler(
    dotted_path: str,
    filter: Literal["all", "public", "private", "dunder"] = "public",
    kind: Literal["all", "functions", "classes", "data", "modules"] | None = None,
    include_inherited: bool = False,
) -> dict:
    """List members of a module or class (like dir())."""
    try:
        from .. import dir

        result = dir(
            dotted_path,
            filter=filter,
            kind=kind,
            include_inherited=include_inherited,
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def exports_handler(dotted_path: str) -> dict:
    """Get the __all__ exports of a module."""
    try:
        from .. import get_exports

        result = get_exports(dotted_path)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def examples_handler(
    dotted_path: str,
    search_paths: list[str] | None = None,
    max_results: int = 10,
) -> dict:
    """Find usage examples of a function/class in tests and examples."""
    try:
        from .. import find_examples

        result = find_examples(
            dotted_path,
            search_paths=search_paths,
            max_results=max_results,
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# Advanced handlers


async def class_hierarchy_handler(
    dotted_path: str,
    include_builtins: bool = False,
    max_depth: int = 10,
) -> dict:
    """Get the inheritance hierarchy of a class."""
    try:
        from .. import get_class_hierarchy

        result = get_class_hierarchy(
            dotted_path,
            include_builtins=include_builtins,
            max_depth=max_depth,
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def type_hints_handler(
    dotted_path: str,
    include_extras: bool = True,
) -> dict:
    """Get detailed type hint information."""
    try:
        from .. import get_type_hints_detailed

        result = get_type_hints_detailed(
            dotted_path,
            include_extras=include_extras,
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def imports_handler(
    dotted_path: str,
    categorize: bool = True,
) -> dict:
    """Get all imports from a module."""
    try:
        from .. import get_imports

        result = get_imports(
            dotted_path,
            categorize=categorize,
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def dependencies_handler(
    dotted_path: str,
    recursive: bool = False,
    max_depth: int = 3,
) -> dict:
    """Get module dependencies."""
    try:
        from .. import get_dependencies

        result = get_dependencies(
            dotted_path,
            recursive=recursive,
            max_depth=max_depth,
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def call_graph_handler(
    dotted_path: str,
    max_depth: int = 2,
    timeout_seconds: int = 10,
    internal_only: bool = True,
) -> dict:
    """Get the call graph of a function or module."""
    try:
        from .. import get_call_graph

        result = get_call_graph(
            dotted_path,
            max_depth=max_depth,
            timeout_seconds=timeout_seconds,
            internal_only=internal_only,
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_api_handler(
    dotted_path: str,
    max_depth: int = 5,
    docstring: bool = False,
    root_only: bool = False,
) -> dict:
    """List the API tree of a module recursively."""
    try:
        from .. import list_api

        df = list_api(
            dotted_path,
            max_depth=max_depth,
            docstring=docstring,
            root_only=root_only,
        )
        return {
            "success": True,
            "api": df.to_dict(orient="records"),
            "count": len(df),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
