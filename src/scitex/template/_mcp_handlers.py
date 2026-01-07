#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/template/_mcp_handlers.py
# ----------------------------------------

"""
MCP Handler implementations for SciTeX Template module.

Provides async handlers for project scaffolding operations:
- list_templates_handler: List available templates
- get_template_info_handler: Get template details
- clone_template_handler: Create project from template
- list_git_strategies_handler: List git strategies
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional


async def list_templates_handler() -> dict:
    """
    List all available SciTeX project templates.

    Returns
    -------
    dict
        Success status and list of templates with basic info
    """
    try:
        from scitex.template import get_available_templates_info

        templates = get_available_templates_info()

        # Simplify for listing
        template_list = [
            {
                "id": t["id"],
                "name": t["name"],
                "description": t["description"],
                "use_case": t["use_case"],
            }
            for t in templates
        ]

        return {
            "success": True,
            "count": len(template_list),
            "templates": template_list,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def get_template_info_handler(template_id: str) -> dict:
    """
    Get detailed information about a specific template.

    Parameters
    ----------
    template_id : str
        Template identifier (research, pip_project, singularity, paper_directory)

    Returns
    -------
    dict
        Success status and template details
    """
    try:
        from scitex.template import get_available_templates_info

        templates = get_available_templates_info()

        # Find the requested template
        template = None
        for t in templates:
            if t["id"] == template_id:
                template = t
                break

        if template is None:
            return {
                "success": False,
                "error": f"Template not found: {template_id}",
                "available_templates": [t["id"] for t in templates],
            }

        return {
            "success": True,
            "template": template,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def clone_template_handler(
    template_id: str,
    project_name: str,
    target_dir: Optional[str] = None,
    git_strategy: str = "child",
    branch: Optional[str] = None,
    tag: Optional[str] = None,
) -> dict:
    """
    Create a new project by cloning a template.

    Parameters
    ----------
    template_id : str
        Template to clone (research, pip_project, singularity, paper_directory)
    project_name : str
        Name for the new project
    target_dir : str, optional
        Parent directory for the project (default: current directory)
    git_strategy : str, optional
        Git initialization strategy (child, parent, origin, none)
    branch : str, optional
        Specific branch to clone
    tag : str, optional
        Specific tag to clone

    Returns
    -------
    dict
        Success status and project path
    """
    try:
        # Map template_id to clone function
        clone_functions = {
            "research": "clone_research",
            "pip_project": "clone_pip_project",
            "singularity": "clone_singularity",
            "paper_directory": "clone_writer_directory",
        }

        if template_id not in clone_functions:
            return {
                "success": False,
                "error": f"Unknown template: {template_id}",
                "available_templates": list(clone_functions.keys()),
            }

        # Import the specific clone function
        import scitex.template as template_module

        clone_func = getattr(template_module, clone_functions[template_id])

        # Build project path
        if target_dir:
            project_path = str(Path(target_dir) / project_name)
        else:
            project_path = project_name

        # Handle git_strategy='none'
        git_strat = None if git_strategy == "none" else git_strategy

        # Run clone in executor (blocking operation)
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            lambda: clone_func(
                project_dir=project_path,
                git_strategy=git_strat,
                branch=branch,
                tag=tag,
            ),
        )

        if success:
            # Resolve the actual path
            resolved_path = Path(project_path)
            if not resolved_path.is_absolute():
                resolved_path = Path.cwd() / resolved_path

            return {
                "success": True,
                "project_path": str(resolved_path),
                "template": template_id,
                "git_strategy": git_strategy,
                "message": f"Successfully created {template_id} project at {resolved_path}",
            }
        else:
            return {
                "success": False,
                "error": f"Failed to clone template: {template_id}",
                "project_name": project_name,
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def list_git_strategies_handler() -> dict:
    """
    List available git initialization strategies.

    Returns
    -------
    dict
        Success status and list of strategies with descriptions
    """
    strategies = [
        {
            "id": "child",
            "name": "Child Repository",
            "description": "Create an isolated git repository in the project directory",
            "recommended": True,
            "preserves_template_history": False,
        },
        {
            "id": "parent",
            "name": "Parent Repository",
            "description": "Use the parent directory's git repository (project becomes part of parent repo)",
            "recommended": False,
            "preserves_template_history": False,
        },
        {
            "id": "origin",
            "name": "Preserve Origin",
            "description": "Keep the template's original git history (maintains connection to template repo)",
            "recommended": False,
            "preserves_template_history": True,
        },
        {
            "id": "none",
            "name": "No Git",
            "description": "Skip git initialization entirely",
            "recommended": False,
            "preserves_template_history": False,
        },
    ]

    return {
        "success": True,
        "count": len(strategies),
        "strategies": strategies,
        "default": "child",
    }


__all__ = [
    "list_templates_handler",
    "get_template_info_handler",
    "clone_template_handler",
    "list_git_strategies_handler",
]

# EOF
