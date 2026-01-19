#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_call_graph.py

"""Call graph analysis using AST with timeout protection."""

from __future__ import annotations

import ast
import inspect
import signal
from contextlib import contextmanager
from pathlib import Path

from ._resolve import get_type_info, resolve_object


class TimeoutError(Exception):
    """Raised when operation times out."""

    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timeout (Unix only)."""

    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    # Only works on Unix
    try:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except (ValueError, AttributeError):
        # Windows or signal not available - no timeout
        yield


def get_call_graph(
    dotted_path: str,
    max_depth: int = 2,
    timeout_seconds: int = 10,
    internal_only: bool = True,
) -> dict:
    """
    Get the call graph of a function or module using static AST analysis.

    Parameters
    ----------
    dotted_path : str
        Dotted path to the function or module
    max_depth : int
        Maximum depth to traverse calls
    timeout_seconds : int
        Timeout in seconds (0 = no timeout)
    internal_only : bool
        Only show calls to functions in the same module

    Returns
    -------
    dict
        calls: list[dict] - Functions this function calls
        called_by: list[dict] - Functions that call this (if module)
        graph: dict - Full call graph tree

    Examples
    --------
    >>> get_call_graph("scitex.audio.speak")
    """
    try:
        if timeout_seconds > 0:
            with timeout(timeout_seconds):
                return _analyze_call_graph(dotted_path, max_depth, internal_only)
        else:
            return _analyze_call_graph(dotted_path, max_depth, internal_only)
    except TimeoutError as e:
        return {
            "success": False,
            "error": str(e),
            "partial": True,
        }


def _analyze_call_graph(
    dotted_path: str,
    max_depth: int,
    internal_only: bool,
) -> dict:
    """Perform the actual call graph analysis."""
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    type_info = get_type_info(obj)

    # Get source file
    try:
        source_file = inspect.getfile(obj)
        source = Path(source_file).read_text()
        tree = ast.parse(source)
    except Exception as e:
        return {
            "success": False,
            "error": f"Cannot parse source: {e}",
            "type_info": type_info,
        }

    # Build function index for the module
    func_index = _build_function_index(tree)

    if inspect.isfunction(obj):
        # Analyze single function
        func_name = obj.__name__
        if func_name not in func_index:
            return {
                "success": False,
                "error": f"Function '{func_name}' not found in source",
                "type_info": type_info,
            }

        calls = _get_function_calls(func_index[func_name], internal_only, func_index)
        called_by = _find_callers(func_name, func_index)

        return {
            "success": True,
            "function": func_name,
            "calls": calls,
            "call_count": len(calls),
            "called_by": called_by,
            "caller_count": len(called_by),
            "type_info": type_info,
        }

    elif inspect.ismodule(obj):
        # Analyze entire module
        graph = {}
        for func_name, func_node in func_index.items():
            calls = _get_function_calls(func_node, internal_only, func_index)
            graph[func_name] = {
                "calls": calls,
                "line": func_node.lineno,
            }

        return {
            "success": True,
            "module": dotted_path,
            "graph": graph,
            "function_count": len(graph),
            "type_info": type_info,
        }

    else:
        return {
            "success": False,
            "error": "Can only analyze functions or modules",
            "type_info": type_info,
        }


def _build_function_index(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    """Build index of all functions in the AST."""
    index = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            index[node.name] = node
    return index


def _get_function_calls(
    func_node: ast.FunctionDef,
    internal_only: bool,
    func_index: dict,
) -> list[dict]:
    """Extract all function calls from a function."""
    calls = []
    seen = set()

    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            call_info = _extract_call_info(node)
            if call_info and call_info["name"] not in seen:
                # Filter to internal only if requested
                if internal_only and call_info["name"] not in func_index:
                    continue
                seen.add(call_info["name"])
                calls.append(call_info)

    return calls


def _extract_call_info(node: ast.Call) -> dict | None:
    """Extract information about a function call."""
    func = node.func

    if isinstance(func, ast.Name):
        # Simple call: func()
        return {
            "name": func.id,
            "type": "function",
            "line": node.lineno,
        }
    elif isinstance(func, ast.Attribute):
        # Method call: obj.method()
        if isinstance(func.value, ast.Name):
            return {
                "name": f"{func.value.id}.{func.attr}",
                "type": "method",
                "object": func.value.id,
                "method": func.attr,
                "line": node.lineno,
            }
        else:
            return {
                "name": func.attr,
                "type": "method",
                "method": func.attr,
                "line": node.lineno,
            }

    return None


def _find_callers(
    func_name: str,
    func_index: dict[str, ast.FunctionDef],
) -> list[dict]:
    """Find all functions that call the given function."""
    callers = []

    for caller_name, caller_node in func_index.items():
        if caller_name == func_name:
            continue

        for node in ast.walk(caller_node):
            if isinstance(node, ast.Call):
                call_info = _extract_call_info(node)
                if call_info and call_info["name"] == func_name:
                    callers.append(
                        {
                            "name": caller_name,
                            "line": caller_node.lineno,
                        }
                    )
                    break

    return callers


def get_function_calls(
    dotted_path: str,
    include_methods: bool = True,
    include_builtins: bool = False,
) -> dict:
    """
    Get just the outgoing calls from a function.

    Simpler version of get_call_graph for quick lookup.

    Parameters
    ----------
    dotted_path : str
        Dotted path to the function
    include_methods : bool
        Include method calls (obj.method())
    include_builtins : bool
        Include builtin function calls

    Returns
    -------
    dict
        calls: list[str] - Names of called functions
    """
    result = get_call_graph(dotted_path, max_depth=1, internal_only=False)

    if not result.get("success"):
        return result

    calls = result.get("calls", [])

    # Filter
    filtered = []
    builtins = {"print", "len", "range", "str", "int", "float", "list", "dict", "set"}

    for call in calls:
        name = call["name"]
        if not include_methods and call.get("type") == "method":
            continue
        if not include_builtins and name in builtins:
            continue
        filtered.append(name)

    return {
        "success": True,
        "function": dotted_path,
        "calls": filtered,
        "call_count": len(filtered),
    }
