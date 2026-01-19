#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_imports.py

"""Import analysis utilities using AST."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from ._resolve import get_type_info, resolve_object


def get_imports(
    dotted_path: str,
    categorize: bool = True,
) -> dict:
    """
    Get all imports from a module's source code using AST.

    Parameters
    ----------
    dotted_path : str
        Dotted path to the module
    categorize : bool
        Group imports by category (stdlib, third-party, local)

    Returns
    -------
    dict
        imports: list[dict] - All imports with details
        categories: dict - Grouped by category (if categorize=True)

    Examples
    --------
    >>> get_imports("scitex.audio")
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

    # Get source file
    try:
        source_file = inspect.getfile(obj)
    except TypeError:
        return {
            "success": False,
            "error": "Cannot get source file (builtin module?)",
            "type_info": type_info,
        }

    # Read and parse source
    try:
        source = Path(source_file).read_text()
        tree = ast.parse(source)
    except Exception as e:
        return {
            "success": False,
            "error": f"Cannot parse source: {e}",
            "type_info": type_info,
        }

    imports = _extract_imports(tree)

    result = {
        "success": True,
        "module": dotted_path,
        "source_file": source_file,
        "imports": imports,
        "import_count": len(imports),
        "type_info": type_info,
    }

    if categorize:
        result["categories"] = _categorize_imports(imports)

    return result


def _extract_imports(tree: ast.AST) -> list[dict]:
    """Extract all imports from an AST."""
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level  # Relative import level

            for alias in node.names:
                imports.append(
                    {
                        "type": "from",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "level": level,
                        "line": node.lineno,
                    }
                )

    return imports


def _categorize_imports(imports: list[dict]) -> dict:
    """Categorize imports into stdlib, third-party, local."""
    import sys

    stdlib_modules = (
        set(sys.stdlib_module_names)
        if hasattr(sys, "stdlib_module_names")
        else _get_stdlib_modules()
    )

    categories = {
        "stdlib": [],
        "third_party": [],
        "local": [],
    }

    for imp in imports:
        module = imp["module"]
        top_level = module.split(".")[0] if module else ""

        # Relative imports are local
        if imp.get("level", 0) > 0:
            categories["local"].append(imp)
        elif top_level in stdlib_modules:
            categories["stdlib"].append(imp)
        else:
            categories["third_party"].append(imp)

    return categories


def _get_stdlib_modules() -> set:
    """Get stdlib module names for Python < 3.10."""
    import pkgutil

    stdlib = set()
    for module in pkgutil.iter_modules():
        if module.name.startswith("_"):
            continue
        try:
            spec = __import__(module.name).__spec__
            if spec and spec.origin:
                if "site-packages" not in spec.origin:
                    stdlib.add(module.name)
        except Exception:
            pass

    # Add common ones that might be missed
    stdlib.update(
        [
            "abc",
            "ast",
            "asyncio",
            "collections",
            "contextlib",
            "dataclasses",
            "datetime",
            "functools",
            "inspect",
            "io",
            "itertools",
            "json",
            "logging",
            "os",
            "pathlib",
            "re",
            "sys",
            "typing",
            "unittest",
            "warnings",
        ]
    )

    return stdlib


def get_dependencies(
    dotted_path: str,
    recursive: bool = False,
    max_depth: int = 3,
) -> dict:
    """
    Get module dependencies (what it imports).

    Parameters
    ----------
    dotted_path : str
        Dotted path to the module
    recursive : bool
        Recursively analyze imported modules
    max_depth : int
        Maximum recursion depth

    Returns
    -------
    dict
        dependencies: list[str] - Direct dependencies
        tree: dict - Dependency tree (if recursive)
    """
    result = get_imports(dotted_path, categorize=True)
    if not result.get("success"):
        return result

    # Get unique module names
    deps = set()
    for imp in result["imports"]:
        module = imp["module"]
        if module:
            deps.add(module.split(".")[0])

    result["dependencies"] = sorted(deps)
    result["dependency_count"] = len(deps)

    if recursive:
        result["tree"] = _build_dep_tree(dotted_path, max_depth, set())

    return result


def _build_dep_tree(
    module_path: str,
    max_depth: int,
    visited: set,
    current_depth: int = 0,
) -> dict:
    """Build dependency tree recursively."""
    if current_depth >= max_depth or module_path in visited:
        return {"module": module_path, "truncated": True}

    visited.add(module_path)

    result = {"module": module_path, "imports": []}

    imports_result = get_imports(module_path, categorize=False)
    if not imports_result.get("success"):
        return result

    for imp in imports_result.get("imports", []):
        module = imp["module"]
        if module and module not in visited:
            top_level = module.split(".")[0]
            # Only recurse into non-stdlib
            if top_level not in _get_stdlib_modules():
                child = _build_dep_tree(module, max_depth, visited, current_depth + 1)
                result["imports"].append(child)

    return result
