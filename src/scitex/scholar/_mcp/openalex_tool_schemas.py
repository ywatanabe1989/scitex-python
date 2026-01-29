#!/usr/bin/env python3
# Timestamp: 2026-01-29
# File: src/scitex/scholar/_mcp/openalex_tool_schemas.py
"""OpenAlex-Local tool schemas for MCP server.

Provides access to 250M+ papers via openalex-local database.
"""

from __future__ import annotations

import mcp.types as types

__all__ = ["get_openalex_tool_schemas"]


def get_openalex_tool_schemas() -> list[types.Tool]:
    """Return OpenAlex-Local tool schemas."""
    return [
        types.Tool(
            name="openalex_search",
            description=(
                "Search OpenAlex database (250M+ papers) via openalex-local. "
                "Fast full-text search with year filtering. Includes citation data."
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
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="openalex_get",
            description="Get a paper by DOI or OpenAlex ID from OpenAlex database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": "DOI of the paper (e.g., '10.1038/nature12373')",
                    },
                    "openalex_id": {
                        "type": "string",
                        "description": "OpenAlex ID (e.g., 'W2100837269')",
                    },
                },
            },
        ),
        types.Tool(
            name="openalex_count",
            description="Count papers matching a search query in OpenAlex database.",
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
            name="openalex_info",
            description="Get OpenAlex database configuration and status.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


# EOF
