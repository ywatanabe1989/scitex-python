#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/_url_finding.py

"""
URL finding mixin for Scholar class.

Provides URL resolution and PDF URL discovery functionality.
"""

from __future__ import annotations

from typing import Any, Dict, List

from scitex import logging

logger = logging.getLogger(__name__)


class URLFindingMixin:
    """Mixin providing URL finding methods."""

    async def _find_urls_for_doi_async(self, doi: str, context) -> Dict[str, Any]:
        """Find all URLs for a DOI (orchestration layer).

        Workflow:
            DOI -> Publisher URL -> PDF URLs -> OpenURL (fallback)

        Args:
            doi: DOI string
            context: Authenticated browser context

        Returns
        -------
            Dictionary with URL information: {
                "url_doi": "https://doi.org/...",
                "url_publisher": "https://publisher.com/...",
                "urls_pdf": [{"url": "...", "source": "zotero_translator"}],
                "url_openurl_resolved": "..." (if fallback used)
            }
        """
        from scitex.scholar.auth.gateway import (
            OpenURLResolver,
            normalize_doi_as_http,
            resolve_publisher_url_by_navigating_to_doi_page,
        )
        from scitex.scholar.url_finder.ScholarURLFinder import ScholarURLFinder

        urls = {"url_doi": normalize_doi_as_http(doi)}

        # Step 1: Resolve publisher URL
        page = await context.new_page()
        try:
            url_publisher = await resolve_publisher_url_by_navigating_to_doi_page(
                doi, page
            )
            urls["url_publisher"] = url_publisher
        finally:
            await page.close()

        # Step 2: Find PDF URLs from publisher URL
        url_finder = ScholarURLFinder(context, config=self.config)
        urls_pdf = []

        if url_publisher:
            urls_pdf = await url_finder.find_pdf_urls(url_publisher)

        # Step 3: Try OpenURL fallback if no PDFs found
        if not urls_pdf:
            openurl_resolver = OpenURLResolver(config=self.config)
            page = await context.new_page()
            try:
                url_openurl_resolved = await openurl_resolver.resolve_doi(doi, page)
                urls["url_openurl_resolved"] = url_openurl_resolved

                if url_openurl_resolved and url_openurl_resolved != "skipped":
                    urls_pdf = await url_finder.find_pdf_urls(url_openurl_resolved)
            finally:
                await page.close()

        urls["urls_pdf"] = self._deduplicate_pdf_urls(urls_pdf) if urls_pdf else []

        return urls

    def _deduplicate_pdf_urls(self, urls_pdf: List[Dict]) -> List[Dict]:
        """Remove duplicate PDF URLs.

        Args:
            urls_pdf: List of PDF URL dicts

        Returns
        -------
            Deduplicated list of PDF URL dicts
        """
        seen = set()
        unique = []
        for pdf in urls_pdf:
            url = pdf.get("url") if isinstance(pdf, dict) else pdf
            if url not in seen:
                seen.add(url)
                unique.append(pdf)
        return unique


# EOF
