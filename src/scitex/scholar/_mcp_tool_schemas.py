#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/scholar/_mcp_tool_schemas.py
# ----------------------------------------

"""Tool schemas for the scitex-scholar MCP server."""

from __future__ import annotations

import mcp.types as types

__all__ = ["get_tool_schemas"]


def get_tool_schemas() -> list[types.Tool]:
    """Return all tool schemas for the Scholar MCP server."""
    return [
        # Search tools
        types.Tool(
            name="search_papers",
            description=(
                "Search for scientific papers across multiple databases "
                "(PubMed, Crossref, Semantic Scholar, etc.)"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Sources to search: pubmed, crossref, semantic_scholar, "
                            "google_scholar, arxiv"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 20,
                    },
                    "year_min": {
                        "type": "integer",
                        "description": "Minimum publication year",
                    },
                    "year_max": {
                        "type": "integer",
                        "description": "Maximum publication year",
                    },
                },
                "required": ["query"],
            },
        ),
        # DOI Resolution
        types.Tool(
            name="resolve_dois",
            description=(
                "Resolve DOIs from paper titles using Crossref API. "
                "Supports resumable operation for large batches."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bibtex_path": {
                        "type": "string",
                        "description": "Path to BibTeX file to resolve DOIs for",
                    },
                    "titles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of paper titles to resolve DOIs for",
                    },
                    "resume": {
                        "type": "boolean",
                        "description": "Resume from previous progress",
                        "default": True,
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name for organizing results",
                    },
                },
            },
        ),
        # BibTeX Enrichment
        types.Tool(
            name="enrich_bibtex",
            description=(
                "Enrich BibTeX entries with metadata: DOIs, abstracts, "
                "citation counts, impact factors"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bibtex_path": {
                        "type": "string",
                        "description": "Path to BibTeX file to enrich",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output path for enriched BibTeX (optional)",
                    },
                    "add_abstracts": {
                        "type": "boolean",
                        "description": "Add missing abstracts",
                        "default": True,
                    },
                    "add_citations": {
                        "type": "boolean",
                        "description": "Add citation counts",
                        "default": True,
                    },
                    "add_impact_factors": {
                        "type": "boolean",
                        "description": "Add journal impact factors",
                        "default": True,
                    },
                },
                "required": ["bibtex_path"],
            },
        ),
        # PDF Download
        types.Tool(
            name="download_pdf",
            description=(
                "Download a PDF for a paper using DOI. Supports multiple strategies: "
                "direct, publisher, open access repositories."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": "DOI of the paper to download",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save PDF",
                        "default": "./pdfs",
                    },
                    "auth_method": {
                        "type": "string",
                        "description": "Authentication method: openathens, shibboleth, none",
                        "enum": ["openathens", "shibboleth", "none"],
                        "default": "none",
                    },
                    "use_browser": {
                        "type": "boolean",
                        "description": "Use browser-based download for paywalled content",
                        "default": False,
                    },
                },
                "required": ["doi"],
            },
        ),
        # Batch PDF Download
        types.Tool(
            name="download_pdfs_batch",
            description=(
                "Download PDFs for multiple papers with progress tracking. "
                "Supports resumable operation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dois": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of DOIs to download",
                    },
                    "bibtex_path": {
                        "type": "string",
                        "description": "Path to BibTeX file (alternative to dois)",
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name for organizing downloads",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save PDFs",
                    },
                    "max_concurrent": {
                        "type": "integer",
                        "description": "Maximum concurrent downloads",
                        "default": 3,
                    },
                    "resume": {
                        "type": "boolean",
                        "description": "Resume from previous progress",
                        "default": True,
                    },
                },
            },
        ),
        # Library Status
        types.Tool(
            name="get_library_status",
            description=(
                "Get status of the paper library: download progress, "
                "missing PDFs, validation status"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name to check (optional)",
                    },
                    "include_details": {
                        "type": "boolean",
                        "description": "Include detailed per-paper status",
                        "default": False,
                    },
                },
            },
        ),
        # Parse BibTeX
        types.Tool(
            name="parse_bibtex",
            description="Parse a BibTeX file and return paper objects",
            inputSchema={
                "type": "object",
                "properties": {
                    "bibtex_path": {
                        "type": "string",
                        "description": "Path to BibTeX file",
                    },
                },
                "required": ["bibtex_path"],
            },
        ),
        # Validate PDFs
        types.Tool(
            name="validate_pdfs",
            description=(
                "Validate PDF files in library for completeness and readability"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name to validate",
                    },
                    "pdf_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific PDF paths to validate",
                    },
                },
            },
        ),
        # OpenURL Resolution
        types.Tool(
            name="resolve_openurls",
            description=(
                "Resolve publisher URLs via OpenURL resolver for institutional access"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dois": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "DOIs to resolve OpenURLs for",
                    },
                    "resolver_url": {
                        "type": "string",
                        "description": "OpenURL resolver URL (uses default if not specified)",
                    },
                    "resume": {
                        "type": "boolean",
                        "description": "Resume from previous progress",
                        "default": True,
                    },
                },
                "required": ["dois"],
            },
        ),
        # Authentication
        types.Tool(
            name="authenticate",
            description=(
                "Authenticate with institutional access (OpenAthens, Shibboleth)"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Authentication method",
                        "enum": ["openathens", "shibboleth"],
                    },
                    "institution": {
                        "type": "string",
                        "description": "Institution identifier (e.g., 'unimelb')",
                    },
                },
                "required": ["method"],
            },
        ),
        # Export to formats
        types.Tool(
            name="export_papers",
            description="Export papers to various formats (BibTeX, RIS, JSON, CSV)",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name to export",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path",
                    },
                    "format": {
                        "type": "string",
                        "description": "Export format",
                        "enum": ["bibtex", "ris", "json", "csv"],
                        "default": "bibtex",
                    },
                    "filter_has_pdf": {
                        "type": "boolean",
                        "description": "Only export papers with downloaded PDFs",
                        "default": False,
                    },
                },
                "required": ["output_path"],
            },
        ),
    ]


# EOF
