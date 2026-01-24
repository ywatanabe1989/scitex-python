#!/usr/bin/env python3
# Timestamp: 2026-01-24
# File: src/scitex/scholar/_mcp/crossref_handlers.py
"""CrossRef-SciTeX handler implementations via crossref-local delegation.

These handlers delegate to crossref-local for fast access to 167M+ papers.
Branded as crossref-scitex to distinguish from official CrossRef API.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

__all__ = [
    "crossref_search_handler",
    "crossref_get_handler",
    "crossref_count_handler",
    "crossref_citations_handler",
    "crossref_info_handler",
]


def _ensure_crossref():
    """Ensure crossref_scitex module is available."""
    try:
        from scitex.scholar import crossref_scitex

        return crossref_scitex
    except ImportError as e:
        raise RuntimeError(
            "crossref-local not installed. Install with: pip install crossref-local"
        ) from e


async def crossref_search_handler(
    query: str,
    limit: int = 20,
    offset: int = 0,
    year_min: int | None = None,
    year_max: int | None = None,
    enrich: bool = False,
) -> dict:
    """Search CrossRef database (167M+ papers) via crossref-local.

    Args:
        query: Search query string (full-text search)
        limit: Maximum number of results (default: 20)
        offset: Number of results to skip for pagination
        year_min: Minimum publication year filter
        year_max: Maximum publication year filter
        enrich: If True, add citation counts and references
    """
    try:
        crossref = _ensure_crossref()
        loop = asyncio.get_running_loop()

        def do_search():
            # Fetch more results for filtering
            fetch_limit = limit * 2 if (year_min or year_max) else limit
            results = crossref.search(query, limit=fetch_limit, offset=offset)

            if enrich:
                results = crossref.enrich(results)

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
                        "citation_count": work.citation_count,
                        "reference_count": work.reference_count,
                        "type": work.type,
                    }
                )
                if len(papers) >= limit:
                    break

            return papers, results.total

        papers, total = await loop.run_in_executor(None, do_search)

        return {
            "success": True,
            "query": query,
            "total": total,
            "count": len(papers),
            "offset": offset,
            "limit": limit,
            "papers": papers,
            "source": "crossref_local",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def crossref_get_handler(
    doi: str,
    include_citations: bool = False,
    include_references: bool = False,
) -> dict:
    """Get a paper by DOI from CrossRef database.

    Args:
        doi: DOI of the paper
        include_citations: Include list of citing DOIs
        include_references: Include list of referenced DOIs
    """
    try:
        crossref = _ensure_crossref()
        loop = asyncio.get_running_loop()

        def do_get():
            work = crossref.get(doi)
            if not work:
                return None

            result = {
                "doi": work.doi,
                "title": work.title,
                "authors": work.authors,
                "year": work.year,
                "journal": work.journal,
                "abstract": work.abstract,
                "citation_count": work.citation_count,
                "reference_count": work.reference_count,
                "type": work.type,
                "publisher": work.publisher,
                "url": work.url,
            }

            if include_citations:
                result["citing_dois"] = crossref.get_citing(doi)

            if include_references:
                result["referenced_dois"] = crossref.get_cited(doi)

            return result

        result = await loop.run_in_executor(None, do_get)

        if result is None:
            return {
                "success": False,
                "error": f"DOI not found: {doi}",
                "doi": doi,
            }

        return {
            "success": True,
            "paper": result,
            "source": "crossref_local",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def crossref_count_handler(query: str) -> dict:
    """Count papers matching a search query.

    Args:
        query: Search query string
    """
    try:
        crossref = _ensure_crossref()
        loop = asyncio.get_running_loop()

        count = await loop.run_in_executor(None, crossref.count, query)

        return {
            "success": True,
            "query": query,
            "count": count,
            "source": "crossref_local",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def crossref_citations_handler(
    doi: str,
    direction: str = "citing",  # "citing", "cited", or "both"
    limit: int = 100,
) -> dict:
    """Get citation relationships for a paper.

    Args:
        doi: DOI of the paper
        direction: "citing" (papers that cite this), "cited" (references), or "both"
        limit: Maximum number of results per direction
    """
    try:
        crossref = _ensure_crossref()
        loop = asyncio.get_running_loop()

        def do_citations():
            result = {"doi": doi}

            if direction in ("citing", "both"):
                citing = crossref.get_citing(doi)
                result["citing_dois"] = citing[:limit]
                result["citing_count"] = len(citing)

            if direction in ("cited", "both"):
                cited = crossref.get_cited(doi)
                result["cited_dois"] = cited[:limit]
                result["cited_count"] = len(cited)

            return result

        result = await loop.run_in_executor(None, do_citations)

        return {
            "success": True,
            **result,
            "direction": direction,
            "source": "crossref_local",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def crossref_info_handler() -> dict:
    """Get information about CrossRef database configuration and status."""
    try:
        crossref = _ensure_crossref()
        loop = asyncio.get_running_loop()

        info = await loop.run_in_executor(None, crossref.info)

        return {
            "success": True,
            "info": info,
            "mode": crossref.get_mode(),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
