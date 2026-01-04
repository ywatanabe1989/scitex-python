#!/usr/bin/env python3
# Timestamp: "2025-12-28 (ywatanabe)"
# File: ./mcp_servers/scitex-analyzer/helpers/docstring_helpers.py

"""Docstring validation helpers."""

import ast
from typing import Dict, List


def validate_docstring_content(
    docstring: str, node: ast.AST, style: str
) -> Dict[str, List]:
    """Validate docstring content based on style.

    Parameters
    ----------
    docstring : str
        The docstring content to validate
    node : ast.AST
        The AST node (function or class)
    style : str
        Docstring style ('numpy' or 'google')

    Returns
    -------
    dict
        Dictionary with 'issues' list
    """
    issues = []

    if isinstance(node, ast.FunctionDef):
        # Check for parameter documentation
        params = [arg.arg for arg in node.args.args if arg.arg != "self"]

        if style == "numpy":
            # Check for Parameters section
            if params and "Parameters" not in docstring:
                issues.append(
                    {
                        "line": node.lineno,
                        "name": node.name,
                        "type": "missing_params",
                        "severity": "medium",
                        "message": "Missing Parameters section in docstring",
                    }
                )

            # Check for Returns section if not None return
            has_return = any(
                isinstance(n, ast.Return) and n.value for n in ast.walk(node)
            )
            if has_return and "Returns" not in docstring:
                issues.append(
                    {
                        "line": node.lineno,
                        "name": node.name,
                        "type": "missing_returns",
                        "severity": "medium",
                        "message": "Missing Returns section in docstring",
                    }
                )

    return {"issues": issues}


def calculate_docstring_coverage(tree: ast.AST) -> float:
    """Calculate percentage of functions/classes with docstrings.

    Parameters
    ----------
    tree : ast.AST
        The parsed AST tree

    Returns
    -------
    float
        Percentage of documented functions/classes (0-100)
    """
    total = 0
    documented = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):  # Skip private methods
                total += 1
                if ast.get_docstring(node):
                    documented += 1

    return (documented / total * 100) if total > 0 else 100.0


# EOF
