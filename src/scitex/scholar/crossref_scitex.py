#!/usr/bin/env python3
# Timestamp: 2026-01-24
# File: src/scitex/scholar/crossref_scitex.py
"""CrossRef-SciTeX: Local CrossRef database integration for scitex.scholar.

This module provides access to the local CrossRef database (167M+ papers)
through the crossref-local package. Branded as "crossref-scitex" to distinguish
from the official CrossRef API. Supports both direct database access and HTTP
API mode for remote servers.

Quick Start
-----------

Search for papers:
    >>> from scitex.scholar import crossref_scitex
    >>> results = crossref_scitex.search("hippocampal sharp wave ripples")
    >>> print(results.total, "papers found")

Get paper by DOI:
    >>> work = crossref_scitex.get("10.1126/science.aax0758")
    >>> print(work.title)

Configuration
-------------

The mode is automatically detected:
- If CROSSREF_LOCAL_DB is set, uses direct database access
- If CROSSREF_LOCAL_API_URL is set, uses HTTP API
- Default: tries localhost:31291 (SciTeX port scheme)

Environment variables (SCITEX_SCHOLAR_CROSSREF_* takes priority):
    SCITEX_SCHOLAR_CROSSREF_DB: Path to local database
    SCITEX_SCHOLAR_CROSSREF_MODE: 'db' or 'http'
    CROSSREF_LOCAL_DB: Path to local database (fallback)
    CROSSREF_LOCAL_API_URL: HTTP API URL (fallback)

SSH tunnel for remote database:
    $ ssh -L 31291:127.0.0.1:31291 your-server
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

# Set environment variables for crossref-local configuration
# SCITEX_SCHOLAR_CROSSREF_* vars take priority in crossref-local's config.py

if TYPE_CHECKING:
    from crossref_local import SearchResult, Work
    from crossref_local.citations import CitationNetwork

__all__ = [
    # Core search/retrieval
    "search",
    "count",
    "get",
    "get_many",
    "exists",
    # Enrichment
    "enrich",
    "enrich_dois",
    # Configuration
    "configure",
    "configure_http",
    "get_mode",
    "info",
    "is_available",
    # Citation functions
    "get_citing",
    "get_cited",
    "get_citation_count",
    # Classes (re-exported)
    "Work",
    "SearchResult",
    "CitationNetwork",
    # Async API
    "aio",
    # Cache module
    "cache",
]


def _ensure_crossref_local():
    """Ensure crossref-local is available."""
    try:
        import crossref_local

        return crossref_local
    except ImportError as e:
        raise ImportError(
            "crossref-local not installed. Install with: pip install crossref-local"
        ) from e


def is_available() -> bool:
    """Check if crossref-local is available and configured.

    Returns
    -------
        True if crossref-local can be used (either DB or HTTP mode)
    """
    try:
        crl = _ensure_crossref_local()
        info_result = crl.info()
        return info_result.get("status") == "ok"
    except Exception:
        return False


# =============================================================================
# Core Search/Retrieval Functions (delegated to crossref-local)
# =============================================================================


def search(
    query: str,
    limit: int = 20,
    offset: int = 0,
    **kwargs: Any,
) -> SearchResult:
    """Search for papers in the CrossRef database.

    Args:
        query: Search query string (full-text search)
        limit: Maximum number of results (default: 20)
        offset: Number of results to skip for pagination
        **kwargs: Additional arguments passed to crossref-local

    Returns
    -------
        SearchResult containing matching papers

    Examples
    --------
        >>> from scitex.scholar import crossref
        >>> results = crossref.search("deep learning")
        >>> for work in results:
        ...     print(work.title)
    """
    crl = _ensure_crossref_local()
    return crl.search(query, limit=limit, offset=offset, **kwargs)


def count(query: str) -> int:
    """Count papers matching a search query.

    Args:
        query: Search query string

    Returns
    -------
        Number of matching papers
    """
    crl = _ensure_crossref_local()
    return crl.count(query)


def get(doi: str) -> Optional[Work]:
    """Get a paper by DOI.

    Args:
        doi: DOI of the paper (e.g., "10.1126/science.aax0758")

    Returns
    -------
        Work object if found, None otherwise

    Examples
    --------
        >>> work = crossref.get("10.1038/nature12373")
        >>> if work:
        ...     print(work.title, work.year)
    """
    crl = _ensure_crossref_local()
    return crl.get(doi)


def get_many(dois: list[str]) -> list[Work]:
    """Get multiple papers by DOI.

    Args:
        dois: List of DOIs

    Returns
    -------
        List of Work objects (None for DOIs not found)
    """
    crl = _ensure_crossref_local()
    return crl.get_many(dois)


def exists(doi: str) -> bool:
    """Check if a DOI exists in the database.

    Args:
        doi: DOI to check

    Returns
    -------
        True if DOI exists
    """
    crl = _ensure_crossref_local()
    return crl.exists(doi)


# =============================================================================
# Enrichment Functions
# =============================================================================


def enrich(results: SearchResult) -> SearchResult:
    """Enrich search results with citation data.

    Args:
        results: SearchResult to enrich

    Returns
    -------
        Enriched SearchResult with citation counts and references
    """
    crl = _ensure_crossref_local()
    return crl.enrich(results)


def enrich_dois(dois: list[str]) -> list[Work]:
    """Get and enrich papers by DOI with citation data.

    Args:
        dois: List of DOIs to enrich

    Returns
    -------
        List of enriched Work objects
    """
    crl = _ensure_crossref_local()
    return crl.enrich_dois(dois)


# =============================================================================
# Citation Functions
# =============================================================================


def get_citing(doi: str) -> list[str]:
    """Get DOIs of papers that cite this paper.

    Args:
        doi: DOI of the paper

    Returns
    -------
        List of DOIs that cite this paper
    """
    crl = _ensure_crossref_local()
    return crl.get_citing(doi)


def get_cited(doi: str) -> list[str]:
    """Get DOIs of papers cited by this paper.

    Args:
        doi: DOI of the paper

    Returns
    -------
        List of DOIs cited by this paper (references)
    """
    crl = _ensure_crossref_local()
    return crl.get_cited(doi)


def get_citation_count(doi: str) -> int:
    """Get the citation count for a paper.

    Args:
        doi: DOI of the paper

    Returns
    -------
        Number of citations
    """
    crl = _ensure_crossref_local()
    return crl.get_citation_count(doi)


# =============================================================================
# Configuration Functions
# =============================================================================


def configure(db_path: str) -> None:
    """Configure direct database access mode.

    Args:
        db_path: Path to the crossref.db file
    """
    crl = _ensure_crossref_local()
    crl.configure(db_path)


def configure_http(api_url: Optional[str] = None) -> None:
    """Configure HTTP API mode.

    Args:
        api_url: API URL (default: http://localhost:31291)

    Example:
        >>> # After setting up SSH tunnel: ssh -L 31291:127.0.0.1:31291 server
        >>> crossref.configure_http()
    """
    crl = _ensure_crossref_local()
    crl.configure_http(api_url)


def get_mode() -> str:
    """Get the current operating mode.

    Returns
    -------
        "db" for direct database access, "http" for API mode
    """
    crl = _ensure_crossref_local()
    return crl.get_mode()


def info() -> dict:
    """Get information about the crossref-local configuration.

    Returns
    -------
        Dict with status, mode, version, database stats, etc.
    """
    crl = _ensure_crossref_local()
    return crl.info()


# =============================================================================
# Re-exported Classes and Modules
# =============================================================================


def __getattr__(name: str):
    """Lazy import for classes and modules."""
    if name == "Work":
        from crossref_local import Work

        return Work
    elif name == "SearchResult":
        from crossref_local import SearchResult

        return SearchResult
    elif name == "CitationNetwork":
        from crossref_local.citations import CitationNetwork

        return CitationNetwork
    elif name == "aio":
        from crossref_local import aio

        return aio
    elif name == "cache":
        from crossref_local import cache

        return cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# EOF
