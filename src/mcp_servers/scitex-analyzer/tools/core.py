#!/usr/bin/env python3
# Timestamp: "2025-12-28 (ywatanabe)"
# File: ./mcp_servers/scitex-analyzer/tools/core.py

"""Core analysis tools for SciTeX analyzer."""

import re
from pathlib import Path
from typing import Any, Dict, List

from ..constants import SCITEX_PATTERNS
from ..helpers import analyze_configs, analyze_patterns, analyze_structure


def register_core_tools(server):
    """Register core analysis tools with the server.

    Parameters
    ----------
    server : ScitexBaseMCPServer
        The server instance to register tools with
    """

    @server.app.tool()
    async def analyze_scitex_project(
        project_path: str, analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze an entire scitex project for patterns and improvements."""

        project = Path(project_path)
        if not project.exists():
            return {"error": f"Project path {project_path} does not exist"}

        # Analyze project structure
        structure_analysis = await analyze_structure(project)

        # Analyze code patterns
        pattern_analysis = await analyze_patterns(project)

        # Analyze configurations
        config_analysis = await analyze_configs(project)

        # Generate recommendations
        recommendations = await _generate_recommendations(
            structure_analysis, pattern_analysis, config_analysis
        )

        return {
            "project_structure": structure_analysis,
            "code_patterns": pattern_analysis,
            "configurations": config_analysis,
            "recommendations": recommendations,
            "summary": {
                "total_files": structure_analysis["total_files"],
                "scitex_compliance": pattern_analysis["compliance_score"],
                "config_consistency": config_analysis["consistency_score"],
                "priority_issues": len(
                    [r for r in recommendations if r.get("priority") == "high"]
                ),
            },
        }

    @server.app.tool()
    async def explain_scitex_pattern(
        code_snippet: str, pattern_type: str = "auto_detect"
    ) -> Dict[str, Any]:
        """Explain scitex patterns in code for learning purposes."""

        # Detect patterns in code
        detected_patterns = []
        for pattern_name, pattern_regex in SCITEX_PATTERNS.items():
            if re.search(pattern_regex, code_snippet):
                detected_patterns.append(pattern_name)

        if not detected_patterns and pattern_type == "auto_detect":
            return {
                "error": "No SciTeX patterns detected in code snippet",
                "suggestion": "Provide a code snippet containing SciTeX usage",
            }

        # Get primary pattern to explain
        primary_pattern = detected_patterns[0] if detected_patterns else pattern_type

        explanations = _get_pattern_explanations()
        pattern_info = explanations.get(
            primary_pattern,
            {
                "pattern_name": f"SciTeX {primary_pattern}",
                "explanation": "This is a SciTeX pattern. Check documentation for details.",
                "benefits": [
                    "Improved reproducibility",
                    "Better code organization",
                ],
                "related_patterns": [],
            },
        )

        return {
            "detected_patterns": detected_patterns,
            "primary_pattern": primary_pattern,
            **pattern_info,
        }

    @server.app.tool()
    async def suggest_scitex_improvements(
        code: str,
        context: str = "research_script",
        focus_areas: List[str] = ["all"],
    ) -> List[Dict[str, Any]]:
        """Suggest specific scitex improvements for code."""

        suggestions = []
        lines = code.split("\n")

        # Check for performance improvements
        if "performance" in focus_areas or "all" in focus_areas:
            for i, line in enumerate(lines):
                if "for" in line and any(
                    op in code for op in ["np.mean", "np.std", "pd.read"]
                ):
                    suggestions.append(
                        {
                            "type": "performance",
                            "line": i + 1,
                            "issue": "Expensive operation in loop",
                            "suggestion": "Use stx.io.cache() to cache results",
                            "example": "result = stx.io.cache('computation_key', expensive_function, args)",
                            "impact": "Can reduce runtime by 50-90% on subsequent runs",
                            "priority": "high",
                        }
                    )

        # Check for reproducibility improvements
        if "reproducibility" in focus_areas or "all" in focus_areas:
            for i, line in enumerate(lines):
                match = re.search(r"=\s*(0\.\d+|[1-9]\d*\.?\d*)\s*(?:#|$)", line)
                if match and not any(
                    skip in line for skip in ["__version__", "range", "len"]
                ):
                    suggestions.append(
                        {
                            "type": "reproducibility",
                            "line": i + 1,
                            "issue": f"Hardcoded value: {match.group(1)}",
                            "current_code": line.strip(),
                            "suggestion": "Extract to CONFIG.PARAMS",
                            "improved_code": f"threshold = CONFIG.PARAMS.THRESHOLD  # {match.group(1)}",
                            "impact": "Makes parameter configurable across experiments",
                            "priority": "medium",
                        }
                    )

        # Check for maintainability improvements
        if "maintainability" in focus_areas or "all" in focus_areas:
            for i, line in enumerate(lines):
                if (
                    line.strip().startswith("def ")
                    and "->" not in line
                    and "self" not in line
                ):
                    suggestions.append(
                        {
                            "type": "maintainability",
                            "line": i + 1,
                            "issue": "Missing return type hint",
                            "suggestion": "Add type hints for better code clarity",
                            "priority": "low",
                        }
                    )

        # Check for SciTeX-specific improvements
        for i, line in enumerate(lines):
            if "pd.read_csv" in line:
                suggestions.append(
                    {
                        "type": "scitex_adoption",
                        "line": i + 1,
                        "issue": "Using pandas instead of scitex for IO",
                        "current_code": line.strip(),
                        "improved_code": line.replace("pd.read_csv", "stx.io.load"),
                        "impact": "Unified IO handling, better error messages",
                        "priority": "medium",
                    }
                )

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))

        return suggestions

    @server.app.tool()
    async def find_scitex_examples(
        pattern: str, context: str = "all"
    ) -> List[Dict[str, str]]:
        """Find examples of scitex usage patterns."""

        examples = {
            "io_save": [
                {
                    "description": "Save DataFrame with symlink",
                    "code": "stx.io.save(df, './results/data.csv', symlink_from_cwd=True)",
                    "context": "data_analysis",
                },
                {
                    "description": "Save figure with automatic data export",
                    "code": "stx.io.save(fig, './figures/plot.png', symlink_from_cwd=True)",
                    "context": "visualization",
                },
            ],
            "plt_subplots": [
                {
                    "description": "Create tracked figure",
                    "code": "fig, ax = stx.plt.subplots(figsize=(10, 6))",
                    "context": "visualization",
                }
            ],
            "config": [
                {
                    "description": "Access configuration parameters",
                    "code": "threshold = CONFIG.PARAMS.SIGNIFICANCE_THRESHOLD",
                    "context": "analysis",
                }
            ],
        }

        pattern_examples = examples.get(pattern, [])
        if context != "all":
            pattern_examples = [e for e in pattern_examples if e["context"] == context]

        return pattern_examples


async def _generate_recommendations(
    structure: Dict[str, Any],
    patterns: Dict[str, Any],
    configs: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Generate recommendations based on analysis."""
    recommendations = []

    # Structure recommendations
    if structure["missing_directories"]:
        recommendations.append(
            {
                "category": "structure",
                "issue": f"Missing standard directories: {', '.join(structure['missing_directories'])}",
                "suggestion": "Create standard project directories for better organization",
                "command": f"mkdir -p {' '.join(structure['missing_directories'])}",
                "priority": "medium",
            }
        )

    # Pattern recommendations
    if patterns["anti_patterns_found"]["absolute_path"] > 0:
        recommendations.append(
            {
                "category": "patterns",
                "issue": f"Found {patterns['anti_patterns_found']['absolute_path']} absolute paths",
                "suggestion": "Convert to relative paths for reproducibility",
                "priority": "high",
            }
        )

    if patterns["anti_patterns_found"]["hardcoded_number"] > 5:
        recommendations.append(
            {
                "category": "patterns",
                "issue": f"Found {patterns['anti_patterns_found']['hardcoded_number']} hardcoded values",
                "suggestion": "Extract to CONFIG.PARAMS for configurability",
                "priority": "medium",
            }
        )

    # Config recommendations
    if configs["missing_standard_configs"]:
        recommendations.append(
            {
                "category": "configuration",
                "issue": f"Missing configs: {', '.join(configs['missing_standard_configs'])}",
                "suggestion": "Create standard configuration files",
                "priority": "high",
            }
        )

    return recommendations


def _get_pattern_explanations() -> Dict[str, Dict[str, Any]]:
    """Get pattern explanations dictionary."""
    return {
        "io_save": {
            "pattern_name": "SciTeX IO Save Pattern",
            "explanation": (
                "stx.io.save() provides unified file saving across 30+ formats. "
                "It automatically creates output directories relative to the script location, "
                "ensuring reproducible file organization."
            ),
            "benefits": [
                "Automatic directory creation - no need for os.makedirs()",
                "Format detection from file extension",
                "Consistent handling across CSV, JSON, NPY, PNG, etc.",
                "Optional symlink creation for easy CWD access",
                "Script-relative output paths for reproducibility",
            ],
            "example": """stx.io.save(data, './results/output.csv', symlink_from_cwd=True)
# Creates: /path/to/script_out/results/output.csv
# Symlink: ./results/output.csv -> /path/to/script_out/results/output.csv""",
            "common_mistakes": [
                "Using absolute paths instead of relative",
                "Forgetting symlink_from_cwd for easy access",
                "Mixing with pandas.to_csv() or numpy.save()",
            ],
            "related_patterns": ["io_load", "io_cache", "config_paths"],
        },
        "plt_subplots": {
            "pattern_name": "SciTeX Plot Data Tracking",
            "explanation": (
                "stx.plt.subplots() wraps matplotlib subplots with automatic data tracking. "
                "Every plot created tracks its data, which is exported to CSV when saved."
            ),
            "benefits": [
                "Automatic CSV export of all plotted data",
                "Perfect reproducibility of figures",
                "Easy data inspection without re-running code",
                "Debugging plot issues with exported data",
                "Sharing raw data alongside figures",
            ],
            "example": """fig, ax = stx.plt.subplots()
ax.plot(x, y, label='Signal')
stx.io.save(fig, './plot.png')
# Creates: plot.png AND plot.csv with the plotted data""",
            "common_mistakes": [
                "Using plt.subplots() instead of stx.plt.subplots()",
                "Using plt.savefig() which doesn't export data",
                "Not realizing CSV is automatically created",
            ],
            "related_patterns": ["set_xyt", "io_save"],
        },
    }


# EOF
