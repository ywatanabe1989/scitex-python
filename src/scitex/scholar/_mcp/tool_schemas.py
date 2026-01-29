#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/scholar/_mcp.tool_schemas.py
# ----------------------------------------
"""Tool schemas for the scitex-scholar MCP server."""

from __future__ import annotations

import mcp.types as types

__all__ = ["get_tool_schemas"]


def get_tool_schemas() -> list[types.Tool]:
    """Return all tool schemas for the Scholar MCP server."""
    base_schemas = [
        # Search tools
        types.Tool(
            name="search_papers",
            description=(
                "Search for scientific papers. Supports local library search and "
                "external databases (CrossRef, Semantic Scholar, PubMed, arXiv, OpenAlex)"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "search_mode": {
                        "type": "string",
                        "description": "Search mode: 'local' (library only), 'external' (online databases), 'both'",
                        "enum": ["local", "external", "both"],
                        "default": "local",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Sources to search: pubmed, crossref, semantic_scholar, "
                            "arxiv, openalex"
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
                "Start SSO login for institutional access (OpenAthens, Shibboleth). "
                "Call without confirm first to check requirements, then with confirm=True to proceed. "
                "Opens a browser window for authentication. Requires environment "
                "variables like SCITEX_SCHOLAR_OPENATHENS_EMAIL to be set."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Authentication method",
                        "enum": ["openathens", "shibboleth", "ezproxy"],
                    },
                    "institution": {
                        "type": "string",
                        "description": "Institution identifier (e.g., 'unimelb')",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force re-authentication even if session exists",
                        "default": False,
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": (
                            "Set to True to proceed with login after reviewing requirements. "
                            "Default False returns requirements check without starting login."
                        ),
                        "default": False,
                    },
                },
                "required": ["method"],
            },
        ),
        types.Tool(
            name="check_auth_status",
            description=(
                "Check current authentication status without starting login. "
                "Returns whether a valid session exists."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Authentication method to check",
                        "enum": ["openathens", "shibboleth", "ezproxy"],
                        "default": "openathens",
                    },
                    "verify_live": {
                        "type": "boolean",
                        "description": "Verify session with remote server (slower but more accurate)",
                        "default": False,
                    },
                },
            },
        ),
        types.Tool(
            name="logout",
            description="Logout from institutional authentication and clear session cache",
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Authentication method to logout from",
                        "enum": ["openathens", "shibboleth", "ezproxy"],
                        "default": "openathens",
                    },
                    "clear_cache": {
                        "type": "boolean",
                        "description": "Also clear cached session files",
                        "default": True,
                    },
                },
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
        # Project Management
        types.Tool(
            name="create_project",
            description="Create a new scholar project for organizing papers",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project to create",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional project description",
                    },
                },
                "required": ["project_name"],
            },
        ),
        types.Tool(
            name="list_projects",
            description="List all scholar projects in the library",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="add_papers_to_project",
            description="Add papers to a project by DOI or from BibTeX file",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Target project name",
                    },
                    "dois": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of DOIs to add",
                    },
                    "bibtex_path": {
                        "type": "string",
                        "description": "Path to BibTeX file with papers to add",
                    },
                },
                "required": ["project"],
            },
        ),
        # PDF Content Parsing
        types.Tool(
            name="parse_pdf_content",
            description=(
                "Parse PDF content to extract text, sections (IMRaD), tables, "
                "images, and metadata. Supports multiple extraction modes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "Direct path to PDF file",
                    },
                    "doi": {
                        "type": "string",
                        "description": "DOI to find PDF in library",
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name to search for PDF",
                    },
                    "mode": {
                        "type": "string",
                        "description": (
                            "Extraction mode: 'text' (plain text), 'sections' (IMRaD), "
                            "'tables', 'images', 'metadata', 'pages', 'scientific', 'full'"
                        ),
                        "enum": [
                            "text",
                            "sections",
                            "tables",
                            "images",
                            "metadata",
                            "pages",
                            "scientific",
                            "full",
                        ],
                        "default": "scientific",
                    },
                    "extract_sections": {
                        "type": "boolean",
                        "description": "Whether to parse IMRaD sections",
                        "default": True,
                    },
                    "extract_tables": {
                        "type": "boolean",
                        "description": "Whether to extract tables",
                        "default": False,
                    },
                    "extract_images": {
                        "type": "boolean",
                        "description": "Whether to extract images",
                        "default": False,
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum pages to process (None = all)",
                    },
                },
            },
        ),
    ]
    from .crossref_tool_schemas import get_crossref_tool_schemas
    from .job_tool_schemas import get_job_tool_schemas
    from .openalex_tool_schemas import get_openalex_tool_schemas

    return (
        base_schemas
        + get_job_tool_schemas()
        + get_crossref_tool_schemas()
        + get_openalex_tool_schemas()
    )


# EOF
