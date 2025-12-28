# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/core/open_access.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/core/open_access.py
# """
# Open Access Detection Module.
# 
# Provides utilities for determining if a paper is open access based on:
# - Known open access sources (arXiv, PMC, bioRxiv, etc.)
# - Unpaywall API lookup
# - Publisher patterns
# - Journal DOAJ status
# """
# 
# from __future__ import annotations
# 
# import re
# from dataclasses import dataclass
# from enum import Enum
# from typing import Optional, List, Dict, Any
# import asyncio
# import aiohttp
# 
# from scitex import logging
# from scitex.scholar.config import ScholarConfig
# 
# logger = logging.getLogger(__name__)
# 
# # Load OA config from default.yaml (single source of truth)
# _config = None
# 
# 
# def _get_config() -> ScholarConfig:
#     """Get or create singleton config instance."""
#     global _config
#     if _config is None:
#         _config = ScholarConfig()
#     return _config
# 
# 
# def _get_oa_sources() -> frozenset:
#     """Get OA sources from config (single source of truth)."""
#     config = _get_config()
#     sources = config.get("OPENACCESS_SOURCES") or []
#     return frozenset(s.lower() for s in sources)
# 
# 
# def _get_oa_journals() -> tuple:
#     """Get OA journal patterns from config (single source of truth)."""
#     config = _get_config()
#     journals = config.get("OPENACCESS_JOURNALS") or []
#     return tuple(j.lower() for j in journals)
# 
# 
# def _get_unpaywall_email() -> str:
#     """Get Unpaywall API email from config."""
#     config = _get_config()
#     return config.get("unpaywall_email") or "research@scitex.io"
# 
# 
# class OAStatus(Enum):
#     """Open Access status categories (aligned with Unpaywall)."""
# 
#     GOLD = "gold"  # Published in OA journal (DOAJ listed)
#     GREEN = "green"  # Available in repository (arXiv, PMC, etc.)
#     HYBRID = "hybrid"  # OA article in subscription journal
#     BRONZE = "bronze"  # Free to read on publisher site, but no license
#     CLOSED = "closed"  # Paywalled
#     UNKNOWN = "unknown"  # Status not determined
# 
# 
# @dataclass
# class OAResult:
#     """Result of open access detection."""
# 
#     is_open_access: bool
#     status: OAStatus
#     oa_url: Optional[str] = None
#     source: Optional[str] = None  # How we determined OA status
#     license: Optional[str] = None
#     confidence: float = 1.0  # 0-1, how confident we are
# 
# 
# # Open Access Sources and Journals are loaded from config/default.yaml
# # These properties provide lazy-loaded access to config values
# # (single source of truth: config/default.yaml → OPENACCESS_SOURCES, OPENACCESS_JOURNALS)
# 
# # arXiv ID patterns
# ARXIV_PATTERNS = [
#     re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$"),  # New format: 2301.12345
#     re.compile(r"^[a-z-]+/\d{7}(v\d+)?$"),  # Old format: hep-th/9901001
#     re.compile(r"^arxiv:\d{4}\.\d{4,5}(v\d+)?$", re.IGNORECASE),
# ]
# 
# 
# def is_arxiv_id(identifier: str) -> bool:
#     """Check if identifier looks like an arXiv ID."""
#     if not identifier:
#         return False
#     identifier = identifier.strip()
#     return any(p.match(identifier) for p in ARXIV_PATTERNS)
# 
# 
# def is_open_access_source(source: str) -> bool:
#     """Check if source is a known open access repository.
# 
#     Sources are loaded from config/default.yaml → OPENACCESS_SOURCES
#     """
#     if not source:
#         return False
#     return source.lower() in _get_oa_sources()
# 
# 
# def is_open_access_journal(journal_name: str, use_cache: bool = True) -> bool:
#     """Check if journal is a known open access journal.
# 
#     Uses three-tier lookup:
#     1. Fast check against config/default.yaml → OPENACCESS_JOURNALS (pattern matching)
#     2. Comprehensive check against cached OpenAlex OA sources (exact match, 62K+ journals)
#     3. Journal normalizer check (handles abbreviations, variants, historical names)
# 
#     Args:
#         journal_name: Journal name to check
#         use_cache: Whether to use OpenAlex cache (default True)
# 
#     Returns:
#         True if journal is known to be Open Access
#     """
#     if not journal_name:
#         return False
# 
#     journal_lower = journal_name.lower()
# 
#     # Tier 1: Fast pattern match from YAML config
#     if any(oa_journal in journal_lower for oa_journal in _get_oa_journals()):
#         return True
# 
#     # Tier 2: Check OpenAlex cache (62K+ OA sources)
#     if use_cache:
#         try:
#             from .oa_cache import is_oa_journal_cached
# 
#             if is_oa_journal_cached(journal_name):
#                 return True
#         except ImportError:
#             pass  # Cache module not available
# 
#     # Tier 3: Use journal normalizer (handles abbreviations, variants)
#     if use_cache:
#         try:
#             from .journal_normalizer import get_journal_normalizer
# 
#             normalizer = get_journal_normalizer()
#             if normalizer.is_open_access(journal_name):
#                 return True
#         except ImportError:
#             pass  # Normalizer module not available
# 
#     return False
# 
# 
# def detect_oa_from_identifiers(
#     doi: Optional[str] = None,
#     arxiv_id: Optional[str] = None,
#     pmcid: Optional[str] = None,
#     source: Optional[str] = None,
#     journal: Optional[str] = None,
#     is_open_access_flag: Optional[bool] = None,
# ) -> OAResult:
#     """
#     Detect open access status from paper identifiers without API calls.
# 
#     This is fast but may miss some OA papers (e.g., hybrid articles).
#     For comprehensive detection, use check_oa_status_async() with Unpaywall.
# 
#     Args:
#         doi: Paper DOI
#         arxiv_id: arXiv identifier
#         pmcid: PubMed Central ID (starts with PMC)
#         source: Source database (arxiv, pmc, biorxiv, etc.)
#         journal: Journal name
#         is_open_access_flag: Pre-existing OA flag from search API
# 
#     Returns:
#         OAResult with detection results
#     """
#     # If we already have an OA flag from a reliable source, trust it
#     if is_open_access_flag is True:
#         return OAResult(
#             is_open_access=True,
#             status=OAStatus.UNKNOWN,  # We don't know the specific type
#             source="api_flag",
#             confidence=0.9,
#         )
# 
#     # arXiv - always open access (GREEN)
#     if arxiv_id and is_arxiv_id(arxiv_id):
#         return OAResult(
#             is_open_access=True,
#             status=OAStatus.GREEN,
#             oa_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
#             source="arxiv",
#             confidence=1.0,
#         )
# 
#     # PMC - always open access (GREEN)
#     if pmcid and pmcid.upper().startswith("PMC"):
#         pmc_num = pmcid[3:] if pmcid.upper().startswith("PMC") else pmcid
#         return OAResult(
#             is_open_access=True,
#             status=OAStatus.GREEN,
#             oa_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_num}/pdf/",
#             source="pmc",
#             confidence=1.0,
#         )
# 
#     # Known OA source
#     if source and is_open_access_source(source):
#         return OAResult(
#             is_open_access=True,
#             status=OAStatus.GREEN
#             if source.lower() in ["arxiv", "pmc", "biorxiv", "medrxiv"]
#             else OAStatus.GOLD,
#             source=f"source_{source}",
#             confidence=0.95,
#         )
# 
#     # Known OA journal
#     if journal and is_open_access_journal(journal):
#         return OAResult(
#             is_open_access=True,
#             status=OAStatus.GOLD,
#             source="oa_journal",
#             confidence=0.85,
#         )
# 
#     # If we have a DOI but no other OA indicators, it's likely paywalled
#     if doi and not arxiv_id and not pmcid:
#         return OAResult(
#             is_open_access=False,
#             status=OAStatus.UNKNOWN,  # Could be hybrid OA, need Unpaywall to confirm
#             source="no_oa_indicators",
#             confidence=0.6,  # Low confidence - could be hybrid OA
#         )
# 
#     # Unknown
#     return OAResult(
#         is_open_access=False,
#         status=OAStatus.UNKNOWN,
#         source="unknown",
#         confidence=0.3,
#     )
# 
# 
# async def check_oa_status_unpaywall(
#     doi: str,
#     email: str = None,
#     timeout: float = 10.0,
# ) -> OAResult:
#     """
#     Check open access status via Unpaywall API.
# 
#     Unpaywall is the authoritative source for OA status detection.
#     Rate limit: 100,000 requests/day with email.
# 
#     Args:
#         doi: Paper DOI (required)
#         email: Email for Unpaywall API (required for polite access)
#         timeout: Request timeout in seconds
# 
#     Returns:
#         OAResult with comprehensive OA information
#     """
#     if not doi:
#         return OAResult(
#             is_open_access=False,
#             status=OAStatus.UNKNOWN,
#             source="no_doi",
#         )
# 
#     # Use config email if not provided
#     if email is None:
#         email = _get_unpaywall_email()
# 
#     # Clean DOI
#     doi = doi.strip()
#     if doi.lower().startswith("https://doi.org/"):
#         doi = doi[16:]
#     elif doi.lower().startswith("doi:"):
#         doi = doi[4:]
# 
#     url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
# 
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.get(
#                 url, timeout=aiohttp.ClientTimeout(total=timeout)
#             ) as resp:
#                 if resp.status == 404:
#                     return OAResult(
#                         is_open_access=False,
#                         status=OAStatus.UNKNOWN,
#                         source="unpaywall_not_found",
#                         confidence=0.5,
#                     )
# 
#                 if resp.status != 200:
#                     logger.warning(f"Unpaywall API error: {resp.status}")
#                     return OAResult(
#                         is_open_access=False,
#                         status=OAStatus.UNKNOWN,
#                         source="unpaywall_error",
#                         confidence=0.0,
#                     )
# 
#                 data = await resp.json()
# 
#                 is_oa = data.get("is_oa", False)
#                 oa_status_str = data.get("oa_status", "closed")
# 
#                 # Map Unpaywall status to our enum
#                 status_map = {
#                     "gold": OAStatus.GOLD,
#                     "green": OAStatus.GREEN,
#                     "hybrid": OAStatus.HYBRID,
#                     "bronze": OAStatus.BRONZE,
#                     "closed": OAStatus.CLOSED,
#                 }
#                 status = status_map.get(oa_status_str, OAStatus.UNKNOWN)
# 
#                 # Get best OA location
#                 oa_url = None
#                 license_str = None
#                 best_oa = data.get("best_oa_location")
#                 if best_oa:
#                     oa_url = best_oa.get("url_for_pdf") or best_oa.get("url")
#                     license_str = best_oa.get("license")
# 
#                 return OAResult(
#                     is_open_access=is_oa,
#                     status=status,
#                     oa_url=oa_url,
#                     source="unpaywall",
#                     license=license_str,
#                     confidence=1.0,
#                 )
# 
#     except asyncio.TimeoutError:
#         logger.warning(f"Unpaywall timeout for DOI: {doi}")
#         return OAResult(
#             is_open_access=False,
#             status=OAStatus.UNKNOWN,
#             source="unpaywall_timeout",
#             confidence=0.0,
#         )
#     except Exception as e:
#         logger.error(f"Unpaywall API error: {e}")
#         return OAResult(
#             is_open_access=False,
#             status=OAStatus.UNKNOWN,
#             source="unpaywall_exception",
#             confidence=0.0,
#         )
# 
# 
# async def check_oa_status_async(
#     doi: Optional[str] = None,
#     arxiv_id: Optional[str] = None,
#     pmcid: Optional[str] = None,
#     source: Optional[str] = None,
#     journal: Optional[str] = None,
#     is_open_access_flag: Optional[bool] = None,
#     use_unpaywall: bool = True,
#     unpaywall_email: str = None,
# ) -> OAResult:
#     """
#     Comprehensive open access detection.
# 
#     First tries fast local detection, then falls back to Unpaywall API
#     if the status is uncertain.
# 
#     Args:
#         doi: Paper DOI
#         arxiv_id: arXiv identifier
#         pmcid: PubMed Central ID
#         source: Source database
#         journal: Journal name
#         is_open_access_flag: Pre-existing OA flag
#         use_unpaywall: Whether to query Unpaywall for uncertain cases
#         unpaywall_email: Email for Unpaywall API
# 
#     Returns:
#         OAResult with best available OA information
#     """
#     # Try fast local detection first
#     local_result = detect_oa_from_identifiers(
#         doi=doi,
#         arxiv_id=arxiv_id,
#         pmcid=pmcid,
#         source=source,
#         journal=journal,
#         is_open_access_flag=is_open_access_flag,
#     )
# 
#     # If we're confident, return immediately
#     if local_result.confidence >= 0.9:
#         return local_result
# 
#     # If we have a DOI and local detection was uncertain, try Unpaywall
#     if use_unpaywall and doi and local_result.confidence < 0.7:
#         unpaywall_result = await check_oa_status_unpaywall(
#             doi=doi,
#             email=unpaywall_email,
#         )
# 
#         # Unpaywall is authoritative if it returns a result
#         if unpaywall_result.confidence > local_result.confidence:
#             return unpaywall_result
# 
#     return local_result
# 
# 
# def check_oa_status(
#     doi: Optional[str] = None,
#     arxiv_id: Optional[str] = None,
#     pmcid: Optional[str] = None,
#     source: Optional[str] = None,
#     journal: Optional[str] = None,
#     is_open_access_flag: Optional[bool] = None,
#     use_unpaywall: bool = False,  # Default to sync-safe behavior
# ) -> OAResult:
#     """
#     Synchronous wrapper for OA detection.
# 
#     By default only uses local detection (no API calls).
#     Set use_unpaywall=True to use Unpaywall API (requires event loop).
#     """
#     if use_unpaywall:
#         try:
#             loop = asyncio.get_event_loop()
#         except RuntimeError:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
# 
#         return loop.run_until_complete(
#             check_oa_status_async(
#                 doi=doi,
#                 arxiv_id=arxiv_id,
#                 pmcid=pmcid,
#                 source=source,
#                 journal=journal,
#                 is_open_access_flag=is_open_access_flag,
#                 use_unpaywall=True,
#             )
#         )
# 
#     return detect_oa_from_identifiers(
#         doi=doi,
#         arxiv_id=arxiv_id,
#         pmcid=pmcid,
#         source=source,
#         journal=journal,
#         is_open_access_flag=is_open_access_flag,
#     )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/core/open_access.py
# --------------------------------------------------------------------------------
