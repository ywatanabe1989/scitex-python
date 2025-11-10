#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/scholar/pipelines/ScholarPipelineMetadataSingle.py

"""
ScholarPipelineMetadataSingle - Single paper metadata enrichment (API-only)

Functionalities:
  - Enriches single paper with metadata using APIs ONLY
  - NO browser automation, NO PDF downloads
  - Fast and lightweight for BibTeX enrichment

Pipeline Steps:
  1. Resolve DOI from title (if needed) - ScholarEngine API
  2. Fetch metadata (citations, abstract, etc.) - ScholarEngine API
  3. Enrich impact factor - ImpactFactorEngine
  4. Return enriched Paper object

Dependencies:
  - API engines only (no playwright/browser)

IO:
  - input: DOI or title string
  - output: Enriched Paper object (metadata only, no PDFs)
"""

import asyncio
from typing import Optional

from scitex import logging
from scitex.scholar.core import Paper
from scitex.scholar.metadata_engines import ScholarEngine
from scitex.scholar.impact_factor import ImpactFactorEngine
from scitex.scholar.config import ScholarConfig

logger = logging.getLogger(__name__)


class ScholarPipelineMetadataSingle:
    """Process single paper for metadata enrichment only (API-based, no browser)."""

    def __init__(self, config: Optional[ScholarConfig] = None):
        """Initialize metadata pipeline.

        Args:
            config: ScholarConfig instance (optional)
        """
        self.name = self.__class__.__name__
        self.config = config or ScholarConfig()
        self.impact_factor_engine = ImpactFactorEngine()

        logger.info(f"{self.name}: Initialized (API-only, no browser)")

    async def enrich_paper_async(
        self,
        paper: Paper,
        force: bool = False,
    ) -> Paper:
        """Enrich a Paper object with metadata using APIs only.

        Args:
            paper: Paper object to enrich
            force: If True, re-fetch even if metadata exists

        Returns:
            Enriched Paper object
        """
        # Use ScholarEngine to get metadata
        engine = ScholarEngine(config=self.config)

        # Build search query
        search_query = {}
        if paper.metadata.id.doi:
            search_query["doi"] = paper.metadata.id.doi
            logger.info(f"{self.name}: Searching by DOI: {paper.metadata.id.doi}")
        elif paper.metadata.basic.title:
            search_query["title"] = paper.metadata.basic.title
            logger.info(
                f"{self.name}: Searching by title: {paper.metadata.basic.title[:50]}..."
            )
        else:
            logger.warning(f"{self.name}: Paper has no DOI or title, skipping")
            return paper

        try:
            # Fetch metadata from APIs
            metadata_dict = await engine.search_async(**search_query)

            if metadata_dict:
                # Merge metadata into paper
                paper = self._merge_metadata(paper, metadata_dict)
                logger.success(
                    f"{self.name}: ✓ Enriched: {paper.metadata.basic.title[:50] if paper.metadata.basic.title else 'No title'}"
                )
            else:
                logger.warning(
                    f"{self.name}: ✗ No metadata found for query: {search_query}"
                )

            # Enrich impact factor if journal is available
            if paper.metadata.publication.journal:
                logger.info(f"{self.name}: Journal found: {paper.metadata.publication.journal}, enriching impact factor...")
                paper = self._enrich_impact_factor(paper)
            else:
                logger.debug(f"{self.name}: No journal name found, skipping impact factor enrichment")

        except Exception as e:
            logger.error(f"{self.name}: Error during enrichment: {e}")

        return paper

    async def enrich_from_doi_or_title_async(
        self,
        doi_or_title: str,
        force: bool = False,
    ) -> Paper:
        """Create and enrich a Paper from DOI or title string.

        Args:
            doi_or_title: DOI or title string
            force: If True, re-fetch even if cached

        Returns:
            Enriched Paper object
        """
        # Create Paper object
        paper = Paper()

        # Check if input is DOI or title
        is_doi = doi_or_title.strip().startswith("10.")

        if is_doi:
            paper.metadata.id.doi = doi_or_title.strip()
            paper.metadata.id.doi_engines = ["user_input"]
        else:
            paper.metadata.basic.title = doi_or_title.strip()

        # Enrich the paper
        return await self.enrich_paper_async(paper, force=force)

    def _merge_metadata(self, paper: Paper, metadata_dict: dict) -> Paper:
        """Merge metadata dictionary from ScholarEngine into Paper object.

        Args:
            paper: Paper object to update
            metadata_dict: Metadata dictionary from ScholarEngine

        Returns:
            Updated Paper object
        """
        # Merge basic info
        if "basic" in metadata_dict:
            basic = metadata_dict["basic"]
            if basic.get("title") and not paper.metadata.basic.title:
                paper.metadata.basic.title = basic["title"]
            if basic.get("abstract"):
                paper.metadata.basic.abstract = basic["abstract"]
            if basic.get("year"):
                paper.metadata.basic.year = basic["year"]
            if basic.get("authors"):
                paper.metadata.basic.authors = basic["authors"]

        # Merge publication info
        if "publication" in metadata_dict:
            pub = metadata_dict["publication"]
            if pub.get("journal"):
                paper.metadata.publication.journal = pub["journal"]
                paper.metadata.publication.journal_engines = pub.get(
                    "journal_engines", []
                )
            if pub.get("short_journal"):
                paper.metadata.publication.short_journal = pub["short_journal"]
            if pub.get("publisher"):
                paper.metadata.publication.publisher = pub["publisher"]
            if pub.get("volume"):
                paper.metadata.publication.volume = pub["volume"]
            if pub.get("issue"):
                paper.metadata.publication.issue = pub["issue"]
            if pub.get("pages"):
                paper.metadata.publication.pages = pub["pages"]

        # Merge IDs
        if "id" in metadata_dict:
            ids = metadata_dict["id"]
            if ids.get("doi") and not paper.metadata.id.doi:
                paper.metadata.id.doi = ids["doi"]
                paper.metadata.id.doi_engines = ids.get("doi_engines", [])
            if ids.get("pmid"):
                paper.metadata.id.pmid = ids["pmid"]
            if ids.get("arxiv_id"):
                paper.metadata.id.arxiv_id = ids["arxiv_id"]
            if ids.get("semantic_scholar_id"):
                paper.metadata.id.semantic_scholar_id = ids["semantic_scholar_id"]

        # Merge citation count
        if "citation_count" in metadata_dict:
            cit = metadata_dict["citation_count"]
            if cit.get("total") is not None:
                paper.metadata.citation_count.total = cit["total"]
                paper.metadata.citation_count.total_engines = cit.get(
                    "total_engines", []
                )
            # Also merge yearly citation counts
            for key, value in cit.items():
                if key.startswith("20") or key.startswith("19"):  # Year keys
                    year_attr = f"y{key}"
                    if hasattr(paper.metadata.citation_count, year_attr):
                        setattr(paper.metadata.citation_count, year_attr, value)

        # Merge URLs (for reference, not for downloading)
        if "url" in metadata_dict:
            urls = metadata_dict["url"]
            if urls.get("landing_page"):
                paper.metadata.url.landing_page = urls["landing_page"]

        return paper

    def _enrich_impact_factor(self, paper: Paper) -> Paper:
        """Enrich paper with journal impact factor.

        Args:
            paper: Paper object with journal name

        Returns:
            Paper object with enriched impact factor metrics
        """
        if not paper.metadata.publication.journal:
            logger.debug(f"{self.name}: No journal in _enrich_impact_factor")
            return paper

        journal_name = paper.metadata.publication.journal
        logger.info(f"{self.name}: Looking up impact factor for: {journal_name}")

        try:
            metrics = self.impact_factor_engine.get_metrics(journal_name)
            logger.info(f"{self.name}: Metrics returned: {metrics}")

            if metrics:
                # Update impact factor in publication metadata
                paper.metadata.publication.impact_factor = metrics.get(
                    "impact_factor"
                )
                paper.metadata.publication.impact_factor_engines = [
                    metrics.get("source", "ImpactFactorEngine")
                ]

                logger.info(
                    f"{self.name}: Impact factor enriched - "
                    f"{paper.metadata.publication.journal}: "
                    f"IF={metrics.get('impact_factor')}, "
                    f"Quartile={metrics.get('quartile')}, "
                    f"Source={metrics.get('source')}"
                )
            else:
                logger.debug(
                    f"{self.name}: No impact factor found for: "
                    f"{paper.metadata.publication.journal}"
                )

        except Exception as e:
            logger.warning(
                f"{self.name}: Failed to enrich impact factor for "
                f"{paper.metadata.publication.journal}: {e}"
            )

        return paper


# EOF
