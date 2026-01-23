#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/scholar/_mcp/all_handlers.py
# ----------------------------------------

"""Combined handlers for unified MCP server.

Re-exports all handlers from both handlers.py and job_handlers.py
for the unified scitex MCP server.
"""

from __future__ import annotations

# Re-export all handlers from main handlers module
from .handlers import (
    add_papers_to_project_handler,
    authenticate_handler,
    check_auth_status_handler,
    create_project_handler,
    download_pdf_handler,
    download_pdfs_batch_handler,
    enrich_bibtex_handler,
    export_papers_handler,
    get_library_status_handler,
    list_projects_handler,
    logout_handler,
    parse_bibtex_handler,
    parse_pdf_content_handler,
    resolve_dois_handler,
    resolve_openurls_handler,
    search_papers_handler,
    validate_pdfs_handler,
)

# Re-export job management handlers
from .job_handlers import (
    cancel_job_handler,
    fetch_papers_handler,
    get_job_result_handler,
    get_job_status_handler,
    list_jobs_handler,
    start_job_handler,
)

__all__ = [
    # Standard handlers
    "search_papers_handler",
    "resolve_dois_handler",
    "enrich_bibtex_handler",
    "download_pdf_handler",
    "download_pdfs_batch_handler",
    "get_library_status_handler",
    "parse_bibtex_handler",
    "validate_pdfs_handler",
    "resolve_openurls_handler",
    "authenticate_handler",
    "check_auth_status_handler",
    "logout_handler",
    "export_papers_handler",
    "create_project_handler",
    "list_projects_handler",
    "add_papers_to_project_handler",
    "parse_pdf_content_handler",
    # Job management handlers
    "fetch_papers_handler",
    "list_jobs_handler",
    "get_job_status_handler",
    "start_job_handler",
    "cancel_job_handler",
    "get_job_result_handler",
]


# EOF
