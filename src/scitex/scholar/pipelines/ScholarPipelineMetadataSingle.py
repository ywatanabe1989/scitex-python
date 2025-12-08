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
from scitex.scholar.utils import DOIValidator

logger = logging.getLogger(__name__)


class ScholarPipelineMetadataSingle:
    """Process single paper for metadata enrichment only (API-based, no browser)."""

    def __init__(
        self, config: Optional[ScholarConfig] = None, validate_dois: bool = False
    ):
        """Initialize metadata pipeline.

        Args:
            config: ScholarConfig instance (optional)
            validate_dois: Whether to validate DOI accessibility (default: False to avoid extra requests)
        """
        self.name = self.__class__.__name__
        self.config = config or ScholarConfig()
        self.impact_factor_engine = ImpactFactorEngine()
        self.validate_dois = validate_dois
        self.doi_validator = DOIValidator() if validate_dois else None

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

        # Build search query with fallback chain: DOI → arXiv ID → Corpus ID → URL → Title
        search_query = {}

        if paper.metadata.id.doi:
            search_query["doi"] = paper.metadata.id.doi
            logger.info(f"{self.name}: Searching by DOI: {paper.metadata.id.doi}")
        elif paper.metadata.id.arxiv_id:
            # Use arXiv ID to generate DOI for arXiv engine
            arxiv_doi = f"10.48550/arxiv.{paper.metadata.id.arxiv_id}"
            search_query["doi"] = arxiv_doi
            logger.info(
                f"{self.name}: Searching by arXiv ID: {paper.metadata.id.arxiv_id} (DOI: {arxiv_doi})"
            )
        elif paper.metadata.id.corpus_id:
            # Pass corpus_id directly to engines (especially SemanticScholarEngine)
            # Also pass URL for URLDOIEngine to extract DOI
            search_query["corpus_id"] = paper.metadata.id.corpus_id
            search_query["url"] = (
                f"https://api.semanticscholar.org/CorpusId:{paper.metadata.id.corpus_id}"
            )
            logger.info(
                f"{self.name}: Searching by Corpus ID: {paper.metadata.id.corpus_id}"
            )
        elif paper.metadata.url.pdfs and len(paper.metadata.url.pdfs) > 0:
            # Try to extract DOI from URL
            url = (
                paper.metadata.url.pdfs[0].get("url")
                if isinstance(paper.metadata.url.pdfs[0], dict)
                else paper.metadata.url.pdfs[0]
            )
            if url:
                search_query["url"] = url
                logger.info(f"{self.name}: Searching by URL: {url[:50]}...")
        elif paper.metadata.basic.title:
            search_query["title"] = paper.metadata.basic.title
            logger.info(
                f"{self.name}: Searching by title: {paper.metadata.basic.title[:50]}..."
            )
        else:
            logger.warning(
                f"{self.name}: Paper has no DOI, Corpus ID, URL or title, skipping"
            )
            return paper

        try:
            # Phase 1: Fetch metadata from APIs
            metadata_dict = await engine.search_async(**search_query)

            if metadata_dict:
                # Log what we got in Phase 1
                phase1_doi = metadata_dict.get("id", {}).get("doi")
                phase1_abstract = metadata_dict.get("basic", {}).get("abstract")
                phase1_citations = metadata_dict.get("citation_count", {}).get("total")

                logger.info(
                    f"{self.name}: Phase 1 retrieved - DOI: {bool(phase1_doi)}, "
                    f"Abstract: {bool(phase1_abstract)}, Citations: {bool(phase1_citations)}"
                )

                # Merge metadata into paper
                paper = self._merge_metadata(paper, metadata_dict)
                logger.success(
                    f"{self.name}: ✓ Enriched: {paper.metadata.basic.title[:50] if paper.metadata.basic.title else 'No title'}"
                )

                # Phase 2: Two-phase enrichment for Corpus ID → DOI → Full Metadata
                # If we got a DOI but no abstract (common with URLDOIEngine from Corpus ID),
                # do a second search using the DOI to get full metadata
                has_doi = bool(paper.metadata.id.doi)
                has_abstract = bool(paper.metadata.basic.abstract)
                was_url_search = "url" in search_query or "corpus_id" in search_query

                if has_doi and not has_abstract and was_url_search:
                    logger.info(
                        f"{self.name}: Phase 2: DOI found but no abstract, searching with DOI: {paper.metadata.id.doi}"
                    )

                    # Search again with the DOI
                    metadata_dict_phase2 = await engine.search_async(
                        doi=paper.metadata.id.doi
                    )

                    if metadata_dict_phase2:
                        # Log what Phase 2 provides
                        phase2_abstract = metadata_dict_phase2.get("basic", {}).get(
                            "abstract"
                        )
                        phase2_citations = metadata_dict_phase2.get(
                            "citation_count", {}
                        ).get("total")

                        # Merge Phase 2 metadata (this will add abstract, citations, etc.)
                        paper = self._merge_metadata(paper, metadata_dict_phase2)

                        # Check what we got
                        has_abstract_now = bool(paper.metadata.basic.abstract)
                        has_citations = bool(paper.metadata.citation_count.total)

                        logger.success(
                            f"{self.name}: Phase 2 complete - Abstract: {has_abstract_now} "
                            f"(+{len(phase2_abstract) if phase2_abstract else 0} chars), "
                            f"Citations: {has_citations} ({phase2_citations if phase2_citations else 0})"
                        )
                    else:
                        logger.warning(
                            f"{self.name}: Phase 2 returned no additional metadata for DOI: {paper.metadata.id.doi}"
                        )

                # Log final enrichment status
                final_status = []
                if not paper.metadata.id.doi:
                    final_status.append("⚠️  NO DOI")
                if not paper.metadata.basic.abstract:
                    final_status.append("⚠️  NO ABSTRACT")
                if not paper.metadata.citation_count.total:
                    final_status.append("⚠️  NO CITATIONS")

                if final_status:
                    logger.warning(
                        f"{self.name}: Missing metadata: {', '.join(final_status)}"
                    )
            else:
                logger.warning(
                    f"{self.name}: ✗ No metadata found for query: {search_query}"
                )
                logger.warning(
                    f"{self.name}: ⚠️  Paper will have: NO DOI, NO ABSTRACT, NO CITATIONS"
                )

            # Enrich impact factor if journal is available
            if paper.metadata.publication.journal:
                logger.info(
                    f"{self.name}: Journal found: {paper.metadata.publication.journal}, enriching impact factor..."
                )
                paper = self._enrich_impact_factor(paper)
            else:
                logger.debug(
                    f"{self.name}: No journal name found, skipping impact factor enrichment"
                )

        except Exception as e:
            import traceback

            logger.error(f"{self.name}: ❌ Error during enrichment: {e}")
            logger.error(f"{self.name}: Traceback: {traceback.format_exc()}")
            logger.warning(
                f"{self.name}: Paper will be returned with minimal enrichment"
            )

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
                doi_candidate = ids["doi"]

                # Validate DOI if enabled
                if self.validate_dois and self.doi_validator:
                    is_valid, message, status_code, resolved_url = (
                        self.doi_validator.validate_doi(doi_candidate)
                    )
                    if is_valid:
                        paper.metadata.id.doi = doi_candidate
                        paper.metadata.id.doi_engines = ids.get("doi_engines", [])
                        logger.debug(
                            f"{self.name}: DOI validated: {doi_candidate} -> {message}"
                        )
                    else:
                        logger.warning(
                            f"{self.name}: DOI validation failed for '{doi_candidate}': {message}"
                        )
                        # Don't set the invalid DOI
                else:
                    # No validation, accept DOI as-is
                    paper.metadata.id.doi = doi_candidate
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
                paper.metadata.publication.impact_factor = metrics.get("impact_factor")
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
