#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/scholar.py
"""Scholar module tools for FastMCP unified server."""

from __future__ import annotations

import json
from typing import Optional


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def register_scholar_tools(mcp) -> None:
    """Register scholar tools with FastMCP server."""

    @mcp.tool()
    async def scholar_search_papers(
        query: str,
        limit: int = 20,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        search_mode: str = "local",
        sources: Optional[list] = None,
    ) -> str:
        """[scholar] Search for scientific papers."""
        from scitex.scholar._mcp.handlers import search_papers_handler

        result = await search_papers_handler(
            query=query,
            limit=limit,
            year_min=year_min,
            year_max=year_max,
            search_mode=search_mode,
            sources=sources,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_resolve_dois(
        titles: Optional[list] = None,
        bibtex_path: Optional[str] = None,
        project: Optional[str] = None,
        resume: bool = True,
    ) -> str:
        """[scholar] Resolve DOIs from paper titles using Crossref API."""
        from scitex.scholar._mcp.handlers import resolve_dois_handler

        result = await resolve_dois_handler(
            titles=titles,
            bibtex_path=bibtex_path,
            project=project,
            resume=resume,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_get_library_status(
        project: Optional[str] = None, include_details: bool = False
    ) -> str:
        """[scholar] Get status of the paper library."""
        from scitex.scholar._mcp.handlers import get_library_status_handler

        result = await get_library_status_handler(
            project=project, include_details=include_details
        )
        return _json(result)

    @mcp.tool()
    async def scholar_create_project(
        project_name: str, description: Optional[str] = None
    ) -> str:
        """[scholar] Create a new scholar project."""
        from scitex.scholar._mcp.handlers import create_project_handler

        result = await create_project_handler(
            project_name=project_name, description=description
        )
        return _json(result)

    @mcp.tool()
    async def scholar_list_projects() -> str:
        """[scholar] List all scholar projects."""
        from scitex.scholar._mcp.handlers import list_projects_handler

        result = await list_projects_handler()
        return _json(result)


# EOF
