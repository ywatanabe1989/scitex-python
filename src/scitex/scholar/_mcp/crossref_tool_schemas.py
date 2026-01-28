#!/usr/bin/env python3
# Timestamp: 2026-01-29
# File: src/scitex/scholar/_mcp/crossref_tool_schemas.py
"""CrossRef-Local tool schemas for MCP server.

Provides access to 167M+ papers via crossref-local database.
"""

from __future__ import annotations

import mcp.types as types

__all__ = ["get_crossref_tool_schemas"]


def get_crossref_tool_schemas() -> list[types.Tool]:
    """Return CrossRef-Local tool schemas."""
    return [
        types.Tool(
            name="crossref_search",
            description=(
                "Search CrossRef database (167M+ papers) via crossref-local. "
                "Fast full-text search with year filtering and citation enrichment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string (full-text search)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 20,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of results to skip for pagination",
                        "default": 0,
                    },
                    "year_min": {
                        "type": "integer",
                        "description": "Minimum publication year filter",
                    },
                    "year_max": {
                        "type": "integer",
                        "description": "Maximum publication year filter",
                    },
                    "enrich": {
                        "type": "boolean",
                        "description": "Add citation counts and references",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="crossref_get",
            description="Get a paper by DOI from CrossRef database with optional citation data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": "DOI of the paper (e.g., '10.1038/nature12373')",
                    },
                    "include_citations": {
                        "type": "boolean",
                        "description": "Include list of DOIs that cite this paper",
                        "default": False,
                    },
                    "include_references": {
                        "type": "boolean",
                        "description": "Include list of DOIs this paper references",
                        "default": False,
                    },
                },
                "required": ["doi"],
            },
        ),
        types.Tool(
            name="crossref_count",
            description="Count papers matching a search query in CrossRef database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="crossref_citations",
            description="Get citation relationships for a paper (citing papers and/or references).",
            inputSchema={
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": "DOI of the paper",
                    },
                    "direction": {
                        "type": "string",
                        "description": "Citation direction: 'citing', 'cited', or 'both'",
                        "enum": ["citing", "cited", "both"],
                        "default": "citing",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results per direction",
                        "default": 100,
                    },
                },
                "required": ["doi"],
            },
        ),
        types.Tool(
            name="crossref_info",
            description="Get CrossRef database configuration and status.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


# EOF
