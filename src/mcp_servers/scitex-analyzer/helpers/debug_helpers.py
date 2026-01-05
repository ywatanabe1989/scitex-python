#!/usr/bin/env python3
# Timestamp: "2025-12-28 (ywatanabe)"
# File: ./mcp_servers/scitex-analyzer/helpers/debug_helpers.py

"""Debugging helpers for script analysis."""

import ast
from typing import Any, Dict, List


async def analyze_script_issues(
    content: str, tree: ast.AST, error_context: str
) -> List[Dict[str, Any]]:
    """Analyze script for common issues.

    Parameters
    ----------
    content : str
        The script content
    tree : ast.AST
        The parsed AST tree
    error_context : str
        Error context from user

    Returns
    -------
    list of dict
        List of issues found
    """
    issues = []

    # Check for missing imports
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")

    # Check for scitex import
    if "scitex" not in imports and "stx" not in content:
        issues.append(
            {
                "type": "missing_import",
                "severity": "high",
                "message": "Missing scitex import",
            }
        )

    # Check for config loading
    if "CONFIG" not in content and "config" in content.lower():
        issues.append(
            {
                "type": "config_issue",
                "severity": "medium",
                "message": "Possible configuration loading issue",
            }
        )

    return issues


async def generate_debug_suggestions(
    issues: List[Dict], content: str
) -> List[Dict[str, Any]]:
    """Generate debugging suggestions based on issues.

    Parameters
    ----------
    issues : list of dict
        List of issues found
    content : str
        The script content

    Returns
    -------
    list of dict
        List of suggestions
    """
    suggestions = []

    for issue in issues:
        if issue["type"] == "missing_import":
            suggestions.append(
                {
                    "issue": issue["message"],
                    "suggestion": "Add: import scitex as stx",
                    "priority": "high",
                }
            )
        elif issue["type"] == "config_issue":
            suggestions.append(
                {
                    "issue": issue["message"],
                    "suggestion": "Add: CONFIG = stx.io.load_configs()",
                    "priority": "medium",
                }
            )

    return suggestions


async def generate_quick_fixes(
    issues: List[Dict], content: str
) -> List[Dict[str, Any]]:
    """Generate quick fixes for common issues.

    Parameters
    ----------
    issues : list of dict
        List of issues found
    content : str
        The script content

    Returns
    -------
    list of dict
        List of quick fixes
    """
    fixes = []

    for issue in issues:
        if issue["type"] == "missing_import":
            fixes.append(
                {
                    "description": "Add SciTeX import",
                    "find": "#!/usr/bin/env python3",
                    "replace": "#!/usr/bin/env python3\\nimport scitex as stx",
                }
            )

    return fixes


async def analyze_error_context(
    error_context: str, content: str
) -> List[Dict[str, Any]]:
    """Analyze error context for specific suggestions.

    Parameters
    ----------
    error_context : str
        Error context from user
    content : str
        The script content

    Returns
    -------
    list of dict
        List of context-specific suggestions
    """
    suggestions = []

    if "ModuleNotFoundError" in error_context:
        if "scitex" in error_context:
            suggestions.append(
                {
                    "error": "SciTeX not installed",
                    "suggestion": "Run: pip install scitex",
                    "priority": "critical",
                }
            )

    if "FileNotFoundError" in error_context:
        suggestions.append(
            {
                "error": "File not found",
                "suggestion": "Check file paths in configuration",
                "priority": "high",
            }
        )

    return suggestions


# EOF
