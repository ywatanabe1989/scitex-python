#!/usr/bin/env python3
"""Code templates for common SciTeX patterns.

Provides ready-to-use code snippets for scripts and modules.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

# Import templates from subdirectory
from ._templates import CODE_TEMPLATES


def get_code_template(
    template_id: str,
    filepath: Optional[str] = None,
    docstring: Optional[str] = None,
) -> str:
    """
    Get a code template by ID.

    Parameters
    ----------
    template_id : str
        Template identifier (session, session-minimal, session-plot, etc.)
    filepath : str, optional
        File path to include in template header.
    docstring : str, optional
        Custom docstring for the template.

    Returns
    -------
    str
        Formatted template content.
    """
    if template_id not in CODE_TEMPLATES:
        available = ", ".join(CODE_TEMPLATES.keys())
        raise ValueError(f"Unknown template: '{template_id}'. Available: {available}")

    template = CODE_TEMPLATES[template_id]
    content = template["content"]

    # Format placeholders
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filepath = filepath or template["filename"]
    docstring = docstring or "Description of this script/module"

    filename = Path(filepath).name
    filename_stem = Path(filepath).stem

    return content.format(
        timestamp=timestamp,
        filepath=filepath,
        filename=filename,
        filename_stem=filename_stem,
        docstring=docstring,
    )


def list_code_templates():
    """
    List all available code templates.

    Returns
    -------
    list[dict]
        List of template info dictionaries.
    """
    return [
        {
            "id": tid,
            "name": info["name"],
            "description": info["description"],
            "filename": info["filename"],
            "usage": info.get("usage", ""),
        }
        for tid, info in CODE_TEMPLATES.items()
    ]


def get_all_templates() -> str:
    """
    Get all templates combined into one string.

    Returns
    -------
    str
        All templates with headers, ordered by priority:
        1. session - Main workflow template
        2. io - File I/O patterns
        3. Others (config, module, etc.)
    """
    # Priority order
    priority_order = [
        # Core templates (priority 1-3)
        "session",
        "io",
        "config",
        # Session variants
        "session-minimal",
        "session-plot",
        "session-stats",
        "module",
        # Module usage templates
        "plt",
        "stats",
        "scholar",
        "audio",
        "capture",
        "diagram",
        "canvas",
        "writer",
    ]

    sections = []
    sections.append("=" * 70)
    sections.append("SCITEX CODE TEMPLATES")
    sections.append("=" * 70)

    for tid in priority_order:
        if tid not in CODE_TEMPLATES:
            continue
        info = CODE_TEMPLATES[tid]
        sections.append("")
        sections.append("-" * 70)
        sections.append(f"TEMPLATE: {tid}")
        sections.append(f"  {info['name']} - {info['description']}")
        sections.append("-" * 70)
        if info.get("usage"):
            sections.append(info["usage"].strip())
        sections.append("")
        sections.append("```python" if not tid == "config" else "```yaml")
        sections.append(info["content"].strip())
        sections.append("```")

    return "\n".join(sections)


__all__ = [
    "get_code_template",
    "list_code_templates",
    "get_all_templates",
    "CODE_TEMPLATES",
]

# EOF
