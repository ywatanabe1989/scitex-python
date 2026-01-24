#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/scholar.py
"""Scholar module tools for FastMCP unified server."""

from __future__ import annotations

import json
from typing import List, Optional


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
        sources: Optional[List[str]] = None,
    ) -> str:
        """[scholar] Search for scientific papers. Supports local library search and external databases (CrossRef, Semantic Scholar, PubMed, arXiv, OpenAlex)."""
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
        titles: Optional[List[str]] = None,
        bibtex_path: Optional[str] = None,
        project: Optional[str] = None,
        resume: bool = True,
    ) -> str:
        """[scholar] Resolve DOIs from paper titles using Crossref API. Supports resumable operation for large batches."""
        from scitex.scholar._mcp.handlers import resolve_dois_handler

        result = await resolve_dois_handler(
            titles=titles,
            bibtex_path=bibtex_path,
            project=project,
            resume=resume,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_enrich_bibtex(
        bibtex_path: str,
        output_path: Optional[str] = None,
        add_abstracts: bool = True,
        add_citations: bool = True,
        add_impact_factors: bool = True,
    ) -> str:
        """[scholar] Enrich BibTeX entries with metadata: DOIs, abstracts, citation counts, impact factors."""
        from scitex.scholar._mcp.handlers import enrich_bibtex_handler

        result = await enrich_bibtex_handler(
            bibtex_path=bibtex_path,
            output_path=output_path,
            add_abstracts=add_abstracts,
            add_citations=add_citations,
            add_impact_factors=add_impact_factors,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_download_pdf(
        doi: str,
        output_dir: str = "./pdfs",
        auth_method: str = "none",
    ) -> str:
        """[scholar] Download a PDF for a paper using DOI. Supports multiple strategies: direct, publisher, open access repositories."""
        from scitex.scholar._mcp.handlers import download_pdf_handler

        result = await download_pdf_handler(
            doi=doi,
            output_dir=output_dir,
            auth_method=auth_method,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_download_pdfs_batch(
        dois: Optional[List[str]] = None,
        bibtex_path: Optional[str] = None,
        project: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_concurrent: int = 3,
        resume: bool = True,
    ) -> str:
        """[scholar] Download PDFs for multiple papers with progress tracking. Supports resumable operation."""
        from scitex.scholar._mcp.handlers import download_pdfs_batch_handler

        result = await download_pdfs_batch_handler(
            dois=dois,
            bibtex_path=bibtex_path,
            project=project,
            output_dir=output_dir,
            max_concurrent=max_concurrent,
            resume=resume,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_get_library_status(
        project: Optional[str] = None,
        include_details: bool = False,
    ) -> str:
        """[scholar] Get status of the paper library: download progress, missing PDFs, validation status."""
        from scitex.scholar._mcp.handlers import get_library_status_handler

        result = await get_library_status_handler(
            project=project,
            include_details=include_details,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_parse_bibtex(bibtex_path: str) -> str:
        """[scholar] Parse a BibTeX file and return paper objects."""
        from scitex.scholar._mcp.handlers import parse_bibtex_handler

        result = await parse_bibtex_handler(bibtex_path=bibtex_path)
        return _json(result)

    @mcp.tool()
    async def scholar_validate_pdfs(
        project: Optional[str] = None,
        pdf_paths: Optional[List[str]] = None,
    ) -> str:
        """[scholar] Validate PDF files in library for completeness and readability."""
        from scitex.scholar._mcp.handlers import validate_pdfs_handler

        result = await validate_pdfs_handler(
            project=project,
            pdf_paths=pdf_paths,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_resolve_openurls(
        dois: List[str],
        resolver_url: Optional[str] = None,
        resume: bool = True,
    ) -> str:
        """[scholar] Resolve publisher URLs via OpenURL resolver for institutional access."""
        from scitex.scholar._mcp.handlers import resolve_openurls_handler

        result = await resolve_openurls_handler(
            dois=dois,
            resolver_url=resolver_url,
            resume=resume,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_authenticate(
        method: str,
        institution: Optional[str] = None,
        force: bool = False,
        confirm: bool = False,
    ) -> str:
        """[scholar] Start SSO login for institutional access (OpenAthens, Shibboleth). Call without confirm first to check requirements, then with confirm=True to proceed."""
        from scitex.scholar._mcp.handlers import authenticate_handler

        result = await authenticate_handler(
            method=method,
            institution=institution,
            force=force,
            confirm=confirm,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_check_auth_status(
        method: str = "openathens",
        verify_live: bool = False,
    ) -> str:
        """[scholar] Check current authentication status without starting login. Returns whether a valid session exists."""
        from scitex.scholar._mcp.handlers import check_auth_status_handler

        result = await check_auth_status_handler(
            method=method,
            verify_live=verify_live,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_logout(
        method: str = "openathens",
        clear_cache: bool = True,
    ) -> str:
        """[scholar] Logout from institutional authentication and clear session cache."""
        from scitex.scholar._mcp.handlers import logout_handler

        result = await logout_handler(
            method=method,
            clear_cache=clear_cache,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_export_papers(
        output_path: str,
        project: Optional[str] = None,
        format: str = "bibtex",
        filter_has_pdf: bool = False,
    ) -> str:
        """[scholar] Export papers to various formats (BibTeX, RIS, JSON, CSV)."""
        from scitex.scholar._mcp.handlers import export_papers_handler

        result = await export_papers_handler(
            output_path=output_path,
            project=project,
            format=format,
            filter_has_pdf=filter_has_pdf,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_create_project(
        project_name: str,
        description: Optional[str] = None,
    ) -> str:
        """[scholar] Create a new scholar project for organizing papers."""
        from scitex.scholar._mcp.handlers import create_project_handler

        result = await create_project_handler(
            project_name=project_name,
            description=description,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_list_projects() -> str:
        """[scholar] List all scholar projects in the library."""
        from scitex.scholar._mcp.handlers import list_projects_handler

        result = await list_projects_handler()
        return _json(result)

    @mcp.tool()
    async def scholar_add_papers_to_project(
        project: str,
        dois: Optional[List[str]] = None,
        bibtex_path: Optional[str] = None,
    ) -> str:
        """[scholar] Add papers to a project by DOI or from BibTeX file."""
        from scitex.scholar._mcp.handlers import add_papers_to_project_handler

        result = await add_papers_to_project_handler(
            project=project,
            dois=dois,
            bibtex_path=bibtex_path,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_parse_pdf_content(
        pdf_path: Optional[str] = None,
        doi: Optional[str] = None,
        project: Optional[str] = None,
        mode: str = "scientific",
        extract_sections: bool = True,
        extract_tables: bool = False,
        extract_images: bool = False,
        max_pages: Optional[int] = None,
    ) -> str:
        """[scholar] Parse PDF content to extract text, sections (IMRaD), tables, images, and metadata."""
        from scitex.scholar._mcp.handlers import parse_pdf_content_handler

        result = await parse_pdf_content_handler(
            pdf_path=pdf_path,
            doi=doi,
            project=project,
            mode=mode,
            extract_sections=extract_sections,
            extract_tables=extract_tables,
            extract_images=extract_images,
            max_pages=max_pages,
        )
        return _json(result)

    # Job management tools (from job_handlers.py)
    @mcp.tool()
    async def scholar_fetch_papers(
        papers: Optional[List[str]] = None,
        bibtex_path: Optional[str] = None,
        project: Optional[str] = None,
        workers: Optional[int] = None,
        browser_mode: str = "stealth",
        chrome_profile: str = "system",
        force: bool = False,
        output: Optional[str] = None,
        async_mode: bool = True,
    ) -> str:
        """[scholar] Fetch papers to your library. Supports async mode (default) which returns immediately with a job_id for tracking."""
        from scitex.scholar._mcp.job_handlers import fetch_papers_handler

        result = await fetch_papers_handler(
            papers=papers,
            bibtex_path=bibtex_path,
            project=project,
            workers=workers,
            browser_mode=browser_mode,
            chrome_profile=chrome_profile,
            force=force,
            output=output,
            async_mode=async_mode,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_list_jobs(
        status: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """[scholar] List all background jobs with their status."""
        from scitex.scholar._mcp.job_handlers import list_jobs_handler

        result = await list_jobs_handler(
            status=status,
            limit=limit,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_get_job_status(job_id: str) -> str:
        """[scholar] Get detailed status of a specific job including progress."""
        from scitex.scholar._mcp.job_handlers import get_job_status_handler

        result = await get_job_status_handler(job_id=job_id)
        return _json(result)

    @mcp.tool()
    async def scholar_start_job(job_id: str) -> str:
        """[scholar] Start a pending job that was submitted with async mode."""
        from scitex.scholar._mcp.job_handlers import start_job_handler

        result = await start_job_handler(job_id=job_id)
        return _json(result)

    @mcp.tool()
    async def scholar_cancel_job(job_id: str) -> str:
        """[scholar] Cancel a running or pending job."""
        from scitex.scholar._mcp.job_handlers import cancel_job_handler

        result = await cancel_job_handler(job_id=job_id)
        return _json(result)

    @mcp.tool()
    async def scholar_get_job_result(job_id: str) -> str:
        """[scholar] Get the result of a completed job."""
        from scitex.scholar._mcp.job_handlers import get_job_result_handler

        result = await get_job_result_handler(job_id=job_id)
        return _json(result)

    # =========================================================================
    # CrossRef Tools (via crossref-local delegation - 167M+ papers)
    # =========================================================================

    @mcp.tool()
    async def scholar_crossref_search(
        query: str,
        limit: int = 20,
        offset: int = 0,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        enrich: bool = False,
    ) -> str:
        """[scholar] Search CrossRef database (167M+ papers) via crossref-local. Fast full-text search with optional citation enrichment."""
        from scitex.scholar._mcp.crossref_handlers import crossref_search_handler

        result = await crossref_search_handler(
            query=query,
            limit=limit,
            offset=offset,
            year_min=year_min,
            year_max=year_max,
            enrich=enrich,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_crossref_get(
        doi: str,
        include_citations: bool = False,
        include_references: bool = False,
    ) -> str:
        """[scholar] Get paper metadata by DOI from CrossRef database."""
        from scitex.scholar._mcp.crossref_handlers import crossref_get_handler

        result = await crossref_get_handler(
            doi=doi,
            include_citations=include_citations,
            include_references=include_references,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_crossref_count(query: str) -> str:
        """[scholar] Count papers matching a query in CrossRef database."""
        from scitex.scholar._mcp.crossref_handlers import crossref_count_handler

        result = await crossref_count_handler(query=query)
        return _json(result)

    @mcp.tool()
    async def scholar_crossref_citations(
        doi: str,
        direction: str = "citing",
        limit: int = 100,
    ) -> str:
        """[scholar] Get citation relationships (citing/cited papers) for a DOI."""
        from scitex.scholar._mcp.crossref_handlers import crossref_citations_handler

        result = await crossref_citations_handler(
            doi=doi,
            direction=direction,
            limit=limit,
        )
        return _json(result)

    @mcp.tool()
    async def scholar_crossref_info() -> str:
        """[scholar] Get CrossRef database configuration and status info."""
        from scitex.scholar._mcp.crossref_handlers import crossref_info_handler

        result = await crossref_info_handler()
        return _json(result)


# EOF
