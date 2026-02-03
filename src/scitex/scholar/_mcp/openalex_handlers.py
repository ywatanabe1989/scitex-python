#!/usr/bin/env python3
# Timestamp: 2026-01-29
# File: src/scitex/scholar/_mcp/openalex_handlers.py
"""OpenAlex-SciTeX handler implementations via openalex-local delegation.

These handlers delegate to openalex-local for fast access to 250M+ papers.
Branded as openalex-scitex to distinguish from official OpenAlex API.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

__all__ = [
    "openalex_search_handler",
    "openalex_get_handler",
    "openalex_count_handler",
    "openalex_info_handler",
]


def _ensure_openalex():
    """Ensure openalex_scitex module is available."""
    try:
        from scitex.scholar import openalex_scitex

        return openalex_scitex
    except ImportError as e:
        raise RuntimeError(
            "openalex-local not installed. Install with: pip install openalex-local"
        ) from e


async def openalex_search_handler(
    query: str,
    limit: int = 20,
    offset: int = 0,
    year_min: int | None = None,
    year_max: int | None = None,
    save_path: str | None = None,
    save_format: str = "json",
) -> dict:
    """Search OpenAlex database (250M+ papers) via openalex-local.

    Args:
        query: Search query string (full-text search)
        limit: Maximum number of results (default: 20)
        offset: Number of results to skip for pagination
        year_min: Minimum publication year filter
        year_max: Maximum publication year filter
        save_path: Optional file path to save results (e.g., "results.json", "papers.bib")
        save_format: Output format for save_path: "text", "json", or "bibtex" (default: "json")
    """
    try:
        openalex = _ensure_openalex()
        loop = asyncio.get_running_loop()

        def do_search():
            # Fetch more results for filtering
            fetch_limit = limit * 2 if (year_min or year_max) else limit
            results = openalex.search(query, limit=fetch_limit, offset=offset)

            papers = []
            for work in results:
                # Apply year filters
                if year_min and work.year and work.year < year_min:
                    continue
                if year_max and work.year and work.year > year_max:
                    continue

                papers.append(
                    {
                        "doi": work.doi,
                        "title": work.title,
                        "authors": work.authors[:10] if work.authors else [],
                        "year": work.year,
                        "journal": work.journal,
                        "abstract": (
                            work.abstract[:500] + "..."
                            if work.abstract and len(work.abstract) > 500
                            else work.abstract
                        ),
                        "citation_count": getattr(work, "citation_count", None),
                        "reference_count": getattr(work, "reference_count", None),
                        "type": getattr(work, "type", None),
                        "openalex_id": getattr(work, "openalex_id", None),
                    }
                )
                if len(papers) >= limit:
                    break

            return papers, getattr(results, "total", len(papers)), results

        papers, total, search_results = await loop.run_in_executor(None, do_search)

        # Save to file if requested
        saved_path = None
        if save_path:
            try:
                from openalex_local import save as oa_save

                saved_path = oa_save(search_results, save_path, format=save_format)
            except Exception as e:
                return {"success": False, "error": f"Failed to save: {e}"}

        result = {
            "success": True,
            "query": query,
            "total": total,
            "count": len(papers),
            "offset": offset,
            "limit": limit,
            "papers": papers,
            "source": "openalex_local",
            "timestamp": datetime.now().isoformat(),
        }

        if saved_path:
            result["saved_to"] = saved_path

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


async def openalex_get_handler(
    doi: str = None,
    openalex_id: str = None,
    save_path: str | None = None,
    save_format: str = "json",
) -> dict:
    """Get a paper by DOI or OpenAlex ID from OpenAlex database.

    Args:
        doi: DOI of the paper (e.g., '10.1038/nature12373')
        openalex_id: OpenAlex ID (e.g., 'W2100837269')
        save_path: Optional file path to save result (e.g., "paper.json", "paper.bib")
        save_format: Output format for save_path: "text", "json", or "bibtex" (default: "json")
    """
    if not doi and not openalex_id:
        return {"success": False, "error": "Must provide either doi or openalex_id"}

    try:
        openalex = _ensure_openalex()
        loop = asyncio.get_running_loop()

        def do_get():
            identifier = doi or openalex_id
            work = openalex.get(identifier)
            if not work:
                return None, None

            result = {
                "doi": work.doi,
                "title": work.title,
                "authors": work.authors,
                "year": work.year,
                "journal": work.journal,
                "abstract": work.abstract,
                "citation_count": getattr(work, "citation_count", None),
                "reference_count": getattr(work, "reference_count", None),
                "type": getattr(work, "type", None),
                "openalex_id": getattr(work, "openalex_id", None),
                "publisher": getattr(work, "publisher", None),
                "url": getattr(work, "url", None),
            }

            return result, work

        result, work_obj = await loop.run_in_executor(None, do_get)

        if result is None:
            identifier = doi or openalex_id
            return {
                "success": False,
                "error": f"Paper not found: {identifier}",
                "identifier": identifier,
            }

        # Save to file if requested
        saved_path = None
        if save_path and work_obj:
            try:
                from openalex_local import save as oa_save

                saved_path = oa_save(work_obj, save_path, format=save_format)
            except Exception as e:
                return {"success": False, "error": f"Failed to save: {e}"}

        response = {
            "success": True,
            "paper": result,
            "source": "openalex_local",
            "timestamp": datetime.now().isoformat(),
        }

        if saved_path:
            response["saved_to"] = saved_path

        return response

    except Exception as e:
        return {"success": False, "error": str(e)}


async def openalex_count_handler(query: str) -> dict:
    """Count papers matching a search query.

    Args:
        query: Search query string
    """
    try:
        openalex = _ensure_openalex()
        loop = asyncio.get_running_loop()

        count = await loop.run_in_executor(None, openalex.count, query)

        return {
            "success": True,
            "query": query,
            "count": count,
            "source": "openalex_local",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def openalex_info_handler() -> dict:
    """Get information about OpenAlex database configuration and status."""
    try:
        openalex = _ensure_openalex()
        loop = asyncio.get_running_loop()

        info = await loop.run_in_executor(None, openalex.info)

        return {
            "success": True,
            "info": info,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
