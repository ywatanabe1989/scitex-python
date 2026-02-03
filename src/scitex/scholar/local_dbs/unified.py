#!/usr/bin/env python3
# Timestamp: 2026-02-04
# File: src/scitex/scholar/local_dbs/unified.py
"""Unified search across CrossRef and OpenAlex local databases.

This module provides a single interface for searching both databases
with automatic deduplication and result merging.

Usage:
    >>> from scitex.scholar.local_dbs.unified import search, get, info
    >>> results = search("hippocampal sharp wave ripples", limit=50)
    >>> print(f"Found {len(results)} papers")
    >>> # Export to different formats
    >>> from scitex.scholar.local_dbs.unified import to_json, to_bibtex, to_text
    >>> print(to_bibtex(results[:5]))
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

__all__ = [
    "search",
    "search_async",
    "get",
    "info",
    "UnifiedWork",
    "UnifiedSearchResult",
    "to_json",
    "to_bibtex",
    "to_text",
    "save",
    "SUPPORTED_FORMATS",
]

# Import save from export module
from .export import SUPPORTED_FORMATS, save

# Try to import both databases
_crossref_available = False
_openalex_available = False

try:
    from crossref_local import Work as CRWork
    from crossref_local import get as cr_get
    from crossref_local import info as cr_info
    from crossref_local import search as cr_search

    _crossref_available = True
except ImportError:
    cr_search = cr_get = cr_info = CRWork = None

try:
    from openalex_local import Work as OAWork
    from openalex_local import get as oa_get
    from openalex_local import info as oa_info
    from openalex_local import search as oa_search

    _openalex_available = True
except ImportError:
    oa_search = oa_get = oa_info = OAWork = None


@dataclass
class UnifiedWork:
    """Unified work representation merging CrossRef and OpenAlex data."""

    doi: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    abstract: Optional[str] = None
    citation_count: Optional[int] = None
    is_open_access: bool = False
    oa_url: Optional[str] = None
    source: str = "unknown"  # "crossref", "openalex", or "merged"

    # Extra fields from OpenAlex
    openalex_id: Optional[str] = None
    concepts: List[Dict] = field(default_factory=list)
    topics: List[Dict] = field(default_factory=list)

    # Extra fields from CrossRef
    issn: Optional[str] = None
    references: List[str] = field(default_factory=list)
    impact_factor: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class UnifiedSearchResult:
    """Container for unified search results."""

    works: List[UnifiedWork]
    total_crossref: int = 0
    total_openalex: int = 0
    duplicates_removed: int = 0
    query: str = ""

    def __len__(self) -> int:
        return len(self.works)

    def __iter__(self):
        return iter(self.works)

    def __getitem__(self, idx):
        return self.works[idx]


def _crossref_work_to_unified(work) -> UnifiedWork:
    """Convert CrossRef work to unified format."""
    return UnifiedWork(
        doi=work.doi,
        title=work.title,
        authors=work.authors if work.authors else [],
        year=work.year,
        journal=work.journal,
        abstract=work.abstract,
        citation_count=work.citation_count,
        source="crossref",
        issn=work.issn if hasattr(work, "issn") else None,
        references=work.references if hasattr(work, "references") else [],
        impact_factor=work.impact_factor if hasattr(work, "impact_factor") else None,
    )


def _openalex_work_to_unified(work) -> UnifiedWork:
    """Convert OpenAlex work to unified format."""
    return UnifiedWork(
        doi=work.doi,
        title=work.title,
        authors=work.authors if work.authors else [],
        year=work.year,
        journal=work.source if hasattr(work, "source") else None,
        abstract=work.abstract,
        citation_count=work.cited_by_count if hasattr(work, "cited_by_count") else None,
        is_open_access=work.is_oa if hasattr(work, "is_oa") else False,
        oa_url=work.oa_url if hasattr(work, "oa_url") else None,
        source="openalex",
        openalex_id=work.openalex_id if hasattr(work, "openalex_id") else None,
        concepts=work.concepts if hasattr(work, "concepts") else [],
        topics=work.topics if hasattr(work, "topics") else [],
    )


def _merge_works(cr_work: UnifiedWork, oa_work: UnifiedWork) -> UnifiedWork:
    """Merge CrossRef and OpenAlex works, preferring more complete data."""
    return UnifiedWork(
        doi=cr_work.doi or oa_work.doi,
        title=cr_work.title or oa_work.title,
        authors=cr_work.authors or oa_work.authors,
        year=cr_work.year or oa_work.year,
        journal=cr_work.journal or oa_work.journal,
        # Prefer OpenAlex abstract (more complete)
        abstract=oa_work.abstract or cr_work.abstract,
        # Prefer higher citation count
        citation_count=max(
            cr_work.citation_count or 0,
            oa_work.citation_count or 0,
        )
        or None,
        is_open_access=oa_work.is_open_access,
        oa_url=oa_work.oa_url,
        source="merged",
        # OpenAlex fields
        openalex_id=oa_work.openalex_id,
        concepts=oa_work.concepts,
        topics=oa_work.topics,
        # CrossRef fields
        issn=cr_work.issn,
        references=cr_work.references,
        impact_factor=cr_work.impact_factor,
    )


def _deduplicate_and_merge(
    cr_works: List[UnifiedWork], oa_works: List[UnifiedWork]
) -> Tuple[List[UnifiedWork], int]:
    """Deduplicate and merge works from both sources."""
    # Index by DOI for fast lookup
    doi_to_cr: Dict[str, UnifiedWork] = {}
    for w in cr_works:
        if w.doi:
            doi_to_cr[w.doi.lower()] = w

    doi_to_oa: Dict[str, UnifiedWork] = {}
    for w in oa_works:
        if w.doi:
            doi_to_oa[w.doi.lower()] = w

    results: List[UnifiedWork] = []
    seen_dois: set = set()
    duplicates = 0

    # Process CrossRef works (merge with OpenAlex if exists)
    for w in cr_works:
        if w.doi:
            doi_lower = w.doi.lower()
            if doi_lower in seen_dois:
                duplicates += 1
                continue
            seen_dois.add(doi_lower)

            if doi_lower in doi_to_oa:
                # Merge with OpenAlex data
                merged = _merge_works(w, doi_to_oa[doi_lower])
                results.append(merged)
                duplicates += 1  # Count the OpenAlex duplicate
            else:
                results.append(w)
        else:
            results.append(w)

    # Add OpenAlex works not in CrossRef
    for w in oa_works:
        if w.doi:
            if w.doi.lower() not in seen_dois:
                results.append(w)
                seen_dois.add(w.doi.lower())
        else:
            results.append(w)

    return results, duplicates


def search(
    query: str,
    limit: int = 50,
    sources: Optional[List[Literal["crossref", "openalex"]]] = None,
    merge_duplicates: bool = True,
    **kwargs,
) -> UnifiedSearchResult:
    """
    Search both CrossRef and OpenAlex databases.

    Args:
        query: Search query string
        limit: Maximum results per source (total may be up to 2x if no dedup)
        sources: Which sources to search. Default: both available sources
        merge_duplicates: Whether to merge works found in both databases
        **kwargs: Additional arguments passed to search functions

    Returns
    -------
        UnifiedSearchResult with merged works
    """
    if sources is None:
        sources = []
        if _crossref_available:
            sources.append("crossref")
        if _openalex_available:
            sources.append("openalex")

    if not sources:
        raise RuntimeError(
            "No search sources available. Install crossref-local or openalex-local"
        )

    cr_works: List[UnifiedWork] = []
    oa_works: List[UnifiedWork] = []

    # Search in parallel using thread pool
    def search_crossref():
        if "crossref" in sources and _crossref_available:
            results = cr_search(query, limit=limit, **kwargs)
            return [_crossref_work_to_unified(w) for w in results]
        return []

    def search_openalex():
        if "openalex" in sources and _openalex_available:
            results = oa_search(query, limit=limit, **kwargs)
            return [_openalex_work_to_unified(w) for w in results]
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        cr_future = executor.submit(search_crossref)
        oa_future = executor.submit(search_openalex)

        try:
            cr_works = cr_future.result(timeout=30)
        except Exception:
            cr_works = []

        try:
            oa_works = oa_future.result(timeout=30)
        except Exception:
            oa_works = []

    # Deduplicate and merge
    duplicates = 0
    if merge_duplicates:
        works, duplicates = _deduplicate_and_merge(cr_works, oa_works)
    else:
        works = cr_works + oa_works

    return UnifiedSearchResult(
        works=works,
        total_crossref=len(cr_works),
        total_openalex=len(oa_works),
        duplicates_removed=duplicates,
        query=query,
    )


async def search_async(
    query: str,
    limit: int = 50,
    sources: Optional[List[Literal["crossref", "openalex"]]] = None,
    merge_duplicates: bool = True,
    **kwargs,
) -> UnifiedSearchResult:
    """Async version of search."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: search(
            query,
            limit=limit,
            sources=sources,
            merge_duplicates=merge_duplicates,
            **kwargs,
        ),
    )


def get(doi: str, sources: Optional[List[str]] = None) -> Optional[UnifiedWork]:
    """
    Get a specific work by DOI from available sources.

    Args:
        doi: DOI to look up
        sources: Which sources to check. Default: all available

    Returns
    -------
        UnifiedWork if found, None otherwise
    """
    if sources is None:
        sources = []
        if _crossref_available:
            sources.append("crossref")
        if _openalex_available:
            sources.append("openalex")

    cr_work = None
    oa_work = None

    if "crossref" in sources and _crossref_available:
        try:
            result = cr_get(doi)
            if result:
                cr_work = _crossref_work_to_unified(result)
        except Exception:
            pass

    if "openalex" in sources and _openalex_available:
        try:
            result = oa_get(doi)
            if result:
                oa_work = _openalex_work_to_unified(result)
        except Exception:
            pass

    if cr_work and oa_work:
        return _merge_works(cr_work, oa_work)
    return cr_work or oa_work


def info() -> Dict[str, Any]:
    """Get status info from all available sources."""
    result = {
        "sources": [],
        "total_works": 0,
    }

    if _crossref_available:
        try:
            cr_info_data = cr_info()
            result["sources"].append(
                {
                    "name": "crossref",
                    "available": True,
                    "info": cr_info_data if isinstance(cr_info_data, dict) else {},
                }
            )
            if isinstance(cr_info_data, dict) and "total_works" in cr_info_data:
                result["total_works"] += cr_info_data["total_works"]
        except Exception as e:
            result["sources"].append(
                {"name": "crossref", "available": False, "error": str(e)}
            )
    else:
        result["sources"].append(
            {"name": "crossref", "available": False, "error": "Not installed"}
        )

    if _openalex_available:
        try:
            oa_info_data = oa_info()
            result["sources"].append(
                {
                    "name": "openalex",
                    "available": True,
                    "info": oa_info_data if isinstance(oa_info_data, dict) else {},
                }
            )
            if isinstance(oa_info_data, dict) and "total_works" in oa_info_data:
                result["total_works"] += oa_info_data["total_works"]
        except Exception as e:
            result["sources"].append(
                {"name": "openalex", "available": False, "error": str(e)}
            )
    else:
        result["sources"].append(
            {"name": "openalex", "available": False, "error": "Not installed"}
        )

    return result


# ============================================================================
# Export Formats
# ============================================================================


def to_json(
    works: Union[List[UnifiedWork], UnifiedSearchResult], indent: int = 2
) -> str:
    """Export works to JSON format."""
    if isinstance(works, UnifiedSearchResult):
        works = works.works
    return json.dumps([w.to_dict() for w in works], indent=indent, ensure_ascii=False)


def to_bibtex(works: Union[List[UnifiedWork], UnifiedSearchResult]) -> str:
    """Export works to BibTeX format."""
    if isinstance(works, UnifiedSearchResult):
        works = works.works

    entries = []
    for i, w in enumerate(works):
        # Generate citation key
        first_author = ""
        if w.authors:
            first_author = (
                w.authors[0].split(",")[0].split()[-1].lower().replace(" ", "")
            )
        year = w.year or "nodate"
        key = f"{first_author}{year}_{i + 1}" if first_author else f"paper_{i + 1}"

        lines = [f"@article{{{key},"]

        if w.title:
            lines.append(f"  title = {{{w.title}}},")
        if w.authors:
            lines.append(f"  author = {{{' and '.join(w.authors)}}},")
        if w.year:
            lines.append(f"  year = {{{w.year}}},")
        if w.journal:
            lines.append(f"  journal = {{{w.journal}}},")
        if w.doi:
            lines.append(f"  doi = {{{w.doi}}},")
        if w.abstract:
            # Escape special LaTeX chars
            abstract = (
                w.abstract.replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
            )
            lines.append(f"  abstract = {{{abstract}}},")

        lines.append("}")
        entries.append("\n".join(lines))

    return "\n\n".join(entries)


def to_text(works: Union[List[UnifiedWork], UnifiedSearchResult]) -> str:
    """Export works to plain text format (one per line)."""
    if isinstance(works, UnifiedSearchResult):
        works = works.works

    lines = []
    for i, w in enumerate(works, 1):
        authors = ", ".join(w.authors[:3]) if w.authors else "Unknown"
        if len(w.authors) > 3:
            authors += " et al."
        title = w.title or "No title"
        year = f"({w.year})" if w.year else ""
        journal = w.journal or ""
        doi = f"DOI: {w.doi}" if w.doi else ""

        line = f"{i}. {authors} {year}. {title}. {journal} {doi}".strip()
        lines.append(line)

    return "\n".join(lines)


# EOF
