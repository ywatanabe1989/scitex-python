#!/usr/bin/env python3
# Timestamp: "2025-12-28 (ywatanabe)"
# File: ./mcp_servers/scitex-analyzer/tools/validation.py

"""Validation tools for SciTeX analyzer."""

import ast
import re
from pathlib import Path
from typing import Any, Dict

from ..constants import IMPORT_ORDER, SEVERITY_PRIORITY
from ..helpers import (
    calculate_docstring_coverage,
    classify_import,
    validate_docstring_content,
)


def register_validation_tools(server):
    """Register validation tools with the server.

    Parameters
    ----------
    server : ScitexBaseMCPServer
        The server instance to register tools with
    """

    @server.app.tool()
    async def validate_comprehensive_compliance(
        project_path: str, strict_mode: bool = False
    ) -> Dict[str, Any]:
        """Perform comprehensive SciTeX guideline compliance validation."""

        project = Path(project_path)
        if not project.exists():
            return {"error": f"Project path {project_path} does not exist"}

        validation_results = {
            "import_order": await _validate_import_order_project(project),
            "docstring_format": await _validate_docstrings_project(project),
            "cross_file_deps": await validate_cross_file_dependencies(str(project)),
            "naming_conventions": await _validate_naming_conventions(project),
            "config_usage": await _validate_config_usage(project),
            "path_handling": await _validate_path_handling(project),
            "framework_compliance": await _validate_framework_compliance(project),
        }

        # Calculate overall score
        total_score = 0
        total_weight = 0
        for category, result in validation_results.items():
            weight = result.get("weight", 1.0)
            score = result.get("score", 0)
            total_score += score * weight
            total_weight += weight

        overall_score = total_score / total_weight if total_weight > 0 else 0

        # Generate report
        violations = []
        for category, result in validation_results.items():
            violations.extend(result.get("violations", []))

        # Sort violations by severity
        violations.sort(
            key=lambda x: SEVERITY_PRIORITY.get(x.get("severity", "low"), 4)
        )

        return {
            "overall_score": round(overall_score, 1),
            "category_scores": {k: v["score"] for k, v in validation_results.items()},
            "total_violations": len(violations),
            "violations_by_severity": {
                "critical": len(
                    [v for v in violations if v.get("severity") == "critical"]
                ),
                "high": len([v for v in violations if v.get("severity") == "high"]),
                "medium": len([v for v in violations if v.get("severity") == "medium"]),
                "low": len([v for v in violations if v.get("severity") == "low"]),
            },
            "detailed_results": validation_results,
            "top_violations": violations[:10],
            "passed": overall_score >= (90 if strict_mode else 70),
        }

    @server.app.tool()
    async def validate_import_order(file_path: str) -> Dict[str, Any]:
        """Validate import order follows SciTeX conventions."""

        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(
                        {
                            "line": node.lineno,
                            "module": node.names[0].name
                            if isinstance(node, ast.Import)
                            else node.module,
                            "type": classify_import(
                                node.names[0].name
                                if isinstance(node, ast.Import)
                                else node.module
                            ),
                        }
                    )

            # Check order
            violations = []
            last_type_index = -1

            for imp in imports:
                current_index = IMPORT_ORDER.index(imp["type"])
                if current_index < last_type_index:
                    violations.append(
                        {
                            "line": imp["line"],
                            "issue": f"{imp['type']} import after {IMPORT_ORDER[last_type_index]}",
                            "module": imp["module"],
                        }
                    )
                last_type_index = max(last_type_index, current_index)

            return {
                "valid": len(violations) == 0,
                "violations": violations,
                "import_count": len(imports),
                "import_breakdown": {
                    t: len([i for i in imports if i["type"] == t]) for t in IMPORT_ORDER
                },
            }

        except Exception as e:
            return {"error": str(e), "valid": False}

    @server.app.tool()
    async def validate_docstring_format(
        file_path: str, style: str = "numpy"
    ) -> Dict[str, Any]:
        """Validate docstring format and completeness."""

        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content)
            issues = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)

                    if not docstring:
                        issues.append(
                            {
                                "line": node.lineno,
                                "name": node.name,
                                "type": "missing",
                                "severity": "high"
                                if not node.name.startswith("_")
                                else "low",
                            }
                        )
                    else:
                        validation = validate_docstring_content(docstring, node, style)
                        if validation["issues"]:
                            issues.extend(validation["issues"])

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "coverage": calculate_docstring_coverage(tree),
                "style": style,
            }

        except Exception as e:
            return {"error": str(e), "valid": False}

    @server.app.tool()
    async def validate_cross_file_dependencies(project_path: str) -> Dict[str, Any]:
        """Validate cross-file dependencies and imports."""

        project = Path(project_path)
        dependency_graph = {}
        issues = []

        # Build dependency graph
        for py_file in project.rglob("*.py"):
            if ".old" in str(py_file):
                continue

            try:
                with open(py_file) as f:
                    content = f.read()

                tree = ast.parse(content)
                file_key = str(py_file.relative_to(project))
                dependency_graph[file_key] = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith("."):
                            dependency_graph[file_key].append(node.module)

            except Exception:
                continue

        # Check for circular dependencies
        for file, deps in dependency_graph.items():
            for dep in deps:
                if dep in dependency_graph and file in dependency_graph.get(dep, []):
                    issues.append(
                        {
                            "type": "circular_import",
                            "files": [file, dep],
                            "severity": "critical",
                        }
                    )

        return {
            "dependency_graph": dependency_graph,
            "issues": issues,
            "total_files": len(dependency_graph),
            "circular_dependencies": len(
                [i for i in issues if i["type"] == "circular_import"]
            ),
        }


async def _validate_import_order_project(project: Path) -> Dict[str, Any]:
    """Validate import order across all files in project."""
    violations = []
    files_checked = 0

    for py_file in project.rglob("*.py"):
        if ".old" in str(py_file):
            continue

        try:
            with open(py_file) as f:
                content = f.read()

            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(
                        {
                            "line": node.lineno,
                            "module": node.names[0].name
                            if isinstance(node, ast.Import)
                            else node.module,
                            "type": classify_import(
                                node.names[0].name
                                if isinstance(node, ast.Import)
                                else node.module
                            ),
                        }
                    )

            last_type_index = -1
            for imp in imports:
                current_index = IMPORT_ORDER.index(imp["type"])
                if current_index < last_type_index:
                    violations.append(
                        {
                            **imp,
                            "file": str(py_file.relative_to(project)),
                            "issue": f"{imp['type']} import after {IMPORT_ORDER[last_type_index]}",
                        }
                    )
                last_type_index = max(last_type_index, current_index)

            files_checked += 1
        except Exception:
            continue

    score = 100 - min(len(violations) * 5, 100)
    return {
        "score": score,
        "violations": violations,
        "files_checked": files_checked,
        "weight": 1.0,
    }


async def _validate_docstrings_project(project: Path) -> Dict[str, Any]:
    """Validate docstrings across all files in project."""
    all_issues = []
    total_coverage = 0
    files_checked = 0

    for py_file in project.rglob("*.py"):
        if ".old" in str(py_file):
            continue

        try:
            with open(py_file) as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if not docstring and not node.name.startswith("_"):
                        all_issues.append(
                            {
                                "file": str(py_file.relative_to(project)),
                                "line": node.lineno,
                                "name": node.name,
                                "type": "missing",
                                "severity": "medium",
                            }
                        )

            total_coverage += calculate_docstring_coverage(tree)
            files_checked += 1
        except Exception:
            continue

    avg_coverage = total_coverage / files_checked if files_checked > 0 else 0
    score = avg_coverage * 0.7 + (100 - min(len(all_issues) * 2, 100)) * 0.3

    return {
        "score": score,
        "violations": all_issues,
        "average_coverage": avg_coverage,
        "files_checked": files_checked,
        "weight": 1.5,
    }


async def _validate_naming_conventions(project: Path) -> Dict[str, Any]:
    """Validate naming conventions."""
    violations = []

    for py_file in project.rglob("*.py"):
        if ".old" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.name.islower() and not node.name.startswith("_"):
                        violations.append(
                            {
                                "file": str(py_file.relative_to(project)),
                                "line": node.lineno,
                                "name": node.name,
                                "issue": "Function name not in snake_case",
                                "severity": "medium",
                            }
                        )
                elif isinstance(node, ast.ClassDef):
                    if not node.name[0].isupper():
                        violations.append(
                            {
                                "file": str(py_file.relative_to(project)),
                                "line": node.lineno,
                                "name": node.name,
                                "issue": "Class name not in PascalCase",
                                "severity": "medium",
                            }
                        )
        except Exception:
            continue

    score = 100 - min(len(violations) * 5, 100)
    return {"score": score, "violations": violations, "weight": 0.5}


async def _validate_config_usage(project: Path) -> Dict[str, Any]:
    """Validate CONFIG usage patterns."""
    violations = []
    good_patterns = 0

    for py_file in project.rglob("*.py"):
        if ".old" in str(py_file):
            continue

        try:
            content = py_file.read_text()

            if re.search(r"['\"][/\\](home|Users|data)[/\\]", content):
                violations.append(
                    {
                        "file": str(py_file.relative_to(project)),
                        "issue": "Hardcoded absolute path",
                        "severity": "high",
                    }
                )

            if "CONFIG" in content:
                good_patterns += len(re.findall(r"CONFIG\.[A-Z_]+\.[A-Z_]+", content))

        except Exception:
            continue

    score = min(100, 70 + good_patterns * 2) - len(violations) * 10
    return {
        "score": max(0, score),
        "violations": violations,
        "good_patterns": good_patterns,
        "weight": 1.5,
    }


async def _validate_path_handling(project: Path) -> Dict[str, Any]:
    """Validate path handling practices."""
    violations = []

    for py_file in project.rglob("*.py"):
        if ".old" in str(py_file):
            continue

        try:
            content = py_file.read_text()

            if "os.path" in content and "pathlib" not in content:
                violations.append(
                    {
                        "file": str(py_file.relative_to(project)),
                        "issue": "Using os.path instead of pathlib",
                        "severity": "low",
                    }
                )

            if re.search(r"['\"].*['\"].*\+.*['\"][/\\]", content):
                violations.append(
                    {
                        "file": str(py_file.relative_to(project)),
                        "issue": "String concatenation for paths",
                        "severity": "medium",
                    }
                )

        except Exception:
            continue

    score = 100 - min(len(violations) * 5, 100)
    return {"score": score, "violations": violations, "weight": 1.0}


async def _validate_framework_compliance(project: Path) -> Dict[str, Any]:
    """Validate SciTeX framework compliance."""
    violations = []
    compliant_scripts = 0

    for py_file in project.rglob("*.py"):
        if ".old" in str(py_file) or "__" in str(py_file):
            continue

        try:
            content = py_file.read_text()

            if "if __name__" in content:
                if "stx.gen.start()" in content:
                    compliant_scripts += 1
                else:
                    violations.append(
                        {
                            "file": str(py_file.relative_to(project)),
                            "issue": "Main script missing stx.gen.start()",
                            "severity": "high",
                        }
                    )

        except Exception:
            continue

    score = 100
    if compliant_scripts == 0 and len(violations) > 0:
        score = 50
    score -= len(violations) * 10

    return {
        "score": max(0, score),
        "violations": violations,
        "compliant_scripts": compliant_scripts,
        "weight": 2.0,
    }


# EOF
