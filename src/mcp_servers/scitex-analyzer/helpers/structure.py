#!/usr/bin/env python3
# Timestamp: "2025-12-28 (ywatanabe)"
# File: ./mcp_servers/scitex-analyzer/helpers/structure.py

"""Project structure analysis helpers."""

import re
from pathlib import Path
from typing import Any, Dict

from ..constants import (
    ANTI_PATTERNS,
    EXPECTED_DIRECTORIES,
    SCITEX_PATTERNS,
    STANDARD_CONFIGS,
)


async def analyze_structure(project_path: Path) -> Dict[str, Any]:
    """Analyze project directory structure.

    Parameters
    ----------
    project_path : Path
        Path to the project directory

    Returns
    -------
    dict
        Structure analysis results
    """
    py_files = list(project_path.rglob("*.py"))
    config_files = list(project_path.rglob("*.yaml")) + list(
        project_path.rglob("*.yml")
    )

    # Check for expected directories
    existing_dirs = [d for d in EXPECTED_DIRECTORIES if (project_path / d).exists()]
    missing_dirs = [d for d in EXPECTED_DIRECTORIES if d not in existing_dirs]

    return {
        "total_files": len(py_files),
        "python_files": len(py_files),
        "config_files": len(config_files),
        "existing_directories": existing_dirs,
        "missing_directories": missing_dirs,
        "structure_score": len(existing_dirs) / len(EXPECTED_DIRECTORIES) * 100,
    }


async def analyze_patterns(project_path: Path) -> Dict[str, Any]:
    """Analyze code patterns in project.

    Parameters
    ----------
    project_path : Path
        Path to the project directory

    Returns
    -------
    dict
        Pattern analysis results with compliance score
    """
    pattern_counts = dict.fromkeys(SCITEX_PATTERNS, 0)
    anti_pattern_counts = dict.fromkeys(ANTI_PATTERNS, 0)
    total_files = 0

    for py_file in project_path.rglob("*.py"):
        if ".old" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            total_files += 1

            # Count patterns
            for name, pattern in SCITEX_PATTERNS.items():
                pattern_counts[name] += len(re.findall(pattern, content))

            # Count anti-patterns
            for name, pattern in ANTI_PATTERNS.items():
                anti_pattern_counts[name] += len(re.findall(pattern, content))

        except Exception:
            continue

    # Calculate compliance score
    total_patterns = sum(pattern_counts.values())
    total_anti_patterns = sum(anti_pattern_counts.values())
    compliance_score = 100
    if total_patterns + total_anti_patterns > 0:
        compliance_score = (
            total_patterns / (total_patterns + total_anti_patterns)
        ) * 100

    return {
        "patterns_found": pattern_counts,
        "anti_patterns_found": anti_pattern_counts,
        "files_analyzed": total_files,
        "compliance_score": round(compliance_score, 1),
        "most_used_pattern": max(pattern_counts, key=pattern_counts.get)
        if pattern_counts
        else None,
        "biggest_issue": max(anti_pattern_counts, key=anti_pattern_counts.get)
        if anti_pattern_counts
        else None,
    }


async def analyze_configs(project_path: Path) -> Dict[str, Any]:
    """Analyze configuration files.

    Parameters
    ----------
    project_path : Path
        Path to the project directory

    Returns
    -------
    dict
        Configuration analysis results with consistency score
    """
    config_files = list(project_path.rglob("*.yaml")) + list(
        project_path.rglob("*.yml")
    )

    configs = {}
    for config_file in config_files:
        if ".old" in str(config_file):
            continue
        configs[str(config_file.relative_to(project_path))] = {
            "exists": True,
            "size": config_file.stat().st_size,
        }

    # Check for standard configs
    missing_configs = [c for c in STANDARD_CONFIGS if not (project_path / c).exists()]

    consistency_score = 100
    if len(STANDARD_CONFIGS) > 0:
        consistency_score = (
            (len(STANDARD_CONFIGS) - len(missing_configs)) / len(STANDARD_CONFIGS)
        ) * 100

    return {
        "config_files": configs,
        "missing_standard_configs": missing_configs,
        "consistency_score": round(consistency_score, 1),
    }


# EOF
