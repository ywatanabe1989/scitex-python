#!/usr/bin/env python3
# Timestamp: "2025-12-28 (ywatanabe)"
# File: ./mcp_servers/scitex-analyzer/tools/documentation.py

"""Documentation and debugging tools for SciTeX analyzer."""

import ast
from pathlib import Path
from typing import Any, Dict

from ..helpers import (
    analyze_error_context,
    analyze_script_issues,
    generate_api_docs,
    generate_config_docs,
    generate_debug_suggestions,
    generate_quick_fixes,
    generate_readme_from_analysis,
    generate_user_guide,
)


def register_documentation_tools(server):
    """Register documentation tools with the server.

    Parameters
    ----------
    server : ScitexBaseMCPServer
        The server instance to register tools with
    """

    @server.app.tool()
    async def debug_scitex_script(
        script_path: str, error_context: str = "", debug_level: str = "standard"
    ) -> Dict[str, Any]:
        """Intelligent debugging assistance for SciTeX scripts."""

        if not Path(script_path).exists():
            return {"error": f"Script not found: {script_path}"}

        # Read script content
        try:
            with open(script_path) as f:
                content = f.read()
        except Exception as e:
            return {"error": f"Cannot read script: {str(e)}"}

        debug_results = {
            "script_path": script_path,
            "debug_level": debug_level,
            "issues_found": [],
            "suggestions": [],
            "quick_fixes": [],
        }

        # Parse code for common issues
        try:
            tree = ast.parse(content)

            # Check for common SciTeX issues
            issues = await analyze_script_issues(content, tree, error_context)
            debug_results["issues_found"] = issues

            # Generate suggestions
            suggestions = await generate_debug_suggestions(issues, content)
            debug_results["suggestions"] = suggestions

            # Generate quick fixes
            quick_fixes = await generate_quick_fixes(issues, content)
            debug_results["quick_fixes"] = quick_fixes

        except SyntaxError as e:
            debug_results["issues_found"].append(
                {
                    "type": "syntax_error",
                    "line": e.lineno,
                    "message": str(e),
                    "severity": "critical",
                }
            )

        # Add context-specific debugging
        if error_context:
            context_suggestions = await analyze_error_context(error_context, content)
            debug_results["context_suggestions"] = context_suggestions

        return debug_results

    @server.app.tool()
    async def generate_scitex_documentation(
        project_path: str,
        doc_type: str = "comprehensive",
        include_examples: bool = True,
    ) -> Dict[str, Any]:
        """Auto-generate project documentation."""

        project = Path(project_path)
        if not project.exists():
            return {"error": f"Project path {project_path} does not exist"}

        # Import analyze_scitex_project from core tools

        # Get project analysis (simplified - would call actual tool)
        analysis = {
            "project_structure": {"total_files": 0, "existing_directories": []},
            "code_patterns": {"compliance_score": 0},
            "configurations": {"consistency_score": 0},
            "recommendations": [],
        }

        docs = {}

        # Generate README if it doesn't exist
        readme_path = project / "README.md"
        if not readme_path.exists():
            readme_content = await generate_readme_from_analysis(analysis, project)
            docs["README.md"] = readme_content

        # Generate API documentation
        if doc_type in ["api", "comprehensive"]:
            api_docs = await generate_api_docs(project)
            docs["docs/API.md"] = api_docs

        # Generate user guide
        if doc_type in ["user_guide", "comprehensive"]:
            user_guide = await generate_user_guide(project, analysis)
            docs["docs/USER_GUIDE.md"] = user_guide

        # Generate configuration documentation
        config_docs = await generate_config_docs(project)
        if config_docs:
            docs["docs/CONFIGURATION.md"] = config_docs

        # Save documentation files
        files_created = []
        for doc_path, content in docs.items():
            full_path = project / doc_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            files_created.append(doc_path)

        return {
            "documentation_generated": True,
            "doc_type": doc_type,
            "files_created": files_created,
            "total_files": len(files_created),
            "project_analysis": analysis.get("summary", {}),
        }


# EOF
