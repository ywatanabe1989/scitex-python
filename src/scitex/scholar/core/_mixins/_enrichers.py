#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/_enrichers.py

"""
Enricher mixin for Scholar class.

Provides paper enrichment functionality including metadata enrichment,
impact factor lookup, and citation count retrieval.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Union

import nest_asyncio

from scitex import logging
from scitex.scholar.impact_factor.ImpactFactorEngine import ImpactFactorEngine

if TYPE_CHECKING:
    from ..Paper import Paper
    from ..Papers import Papers

logger = logging.getLogger(__name__)


class EnricherMixin:
    """Mixin providing paper enrichment methods."""

    async def enrich_papers_async(self, papers: Papers) -> Papers:
        """Async version of enrich_papers for use in async contexts.

        Args:
            papers: Papers collection to enrich.

        Returns
        -------
            Enriched Papers collection
        """
        from ..Papers import Papers

        enriched_list = []

        for paper in papers:
            try:
                results = await self._scholar_engine.search_async(
                    title=paper.metadata.basic.title,
                    year=paper.metadata.basic.year,
                    authors=(
                        paper.metadata.basic.authors[0]
                        if paper.metadata.basic.authors
                        else None
                    ),
                )

                enriched_paper = self._merge_enrichment_data(paper, results)
                enriched_list.append(enriched_paper)
                title = paper.metadata.basic.title or "No title"
                logger.info(f"{self.name}: Enriched: {title[:50]}...")

            except Exception as e:
                title = paper.metadata.basic.title or "No title"
                logger.warning(
                    f"{self.name}: Failed to enrich paper '{title[:50]}...': {e}"
                )
                enriched_list.append(paper)

        enriched_papers = Papers(enriched_list, project=self.project)

        if self.config.resolve("enrich_impact_factors", None, True):
            enriched_papers = self._enrich_impact_factors(enriched_papers)

        return enriched_papers

    def enrich_papers(
        self, papers: Optional[Papers] = None
    ) -> Union[Papers, Dict[str, int]]:
        """Enrich papers with metadata from multiple sources.

        Args:
            papers: Papers collection to enrich. If None, enriches all papers
                   in current project.

        Returns
        -------
            - If papers provided: Returns enriched Papers collection
            - If no papers: Returns dict with enrichment statistics for project
        """
        from ..Papers import Papers

        if papers is None:
            return self._enrich_current_project()

        enriched_list = []
        nest_asyncio.apply()

        for paper in papers:
            try:
                results = asyncio.run(
                    self._scholar_engine.search_async(
                        title=paper.metadata.basic.title,
                        year=paper.metadata.basic.year,
                        authors=(
                            paper.metadata.basic.authors[0]
                            if paper.metadata.basic.authors
                            else None
                        ),
                    )
                )

                enriched_paper = self._merge_enrichment_data(paper, results)
                enriched_list.append(enriched_paper)
                title = paper.metadata.basic.title or "No title"
                logger.info(f"{self.name}: Enriched: {title[:50]}...")

            except Exception as e:
                title = paper.metadata.basic.title or "No title"
                logger.warning(
                    f"{self.name}: Failed to enrich paper '{title[:50]}...': {e}"
                )
                enriched_list.append(paper)

        enriched_papers = Papers(enriched_list, project=self.project)

        if self.config.resolve("enrich_impact_factors", None, True):
            enriched_papers = self._enrich_impact_factors(enriched_papers)

        return enriched_papers

    def _enrich_impact_factors(self, papers: Papers) -> Papers:
        """Add journal impact factors to papers.

        Args:
            papers: Papers collection to enrich with impact factors

        Returns
        -------
            Papers collection with impact factors added where available
        """
        try:
            jcr_engine = ImpactFactorEngine()
            papers = jcr_engine.enrich_papers(papers)
            return papers
        except Exception as e:
            logger.debug(
                f"{self.name}: JCR engine unavailable: {e}, "
                "falling back to calculation method"
            )
        return papers

    def _merge_enrichment_data(self, paper: Paper, results: Dict) -> Paper:
        """Merge enrichment results into paper object.

        Creates a new Paper object with merged data to avoid modifying the original.
        """
        enriched = deepcopy(paper)

        if not results:
            return enriched

        # ID section
        if "id" in results:
            if results["id"].get("doi") and not enriched.metadata.id.doi:
                enriched.metadata.set_doi(results["id"]["doi"])
            if results["id"].get("pmid") and not enriched.metadata.id.pmid:
                enriched.metadata.id.pmid = results["id"]["pmid"]
            if results["id"].get("arxiv_id") and not enriched.metadata.id.arxiv_id:
                enriched.metadata.id.arxiv_id = results["id"]["arxiv_id"]

        # Basic metadata section
        if "basic" in results:
            if results["basic"].get("abstract"):
                enriched.metadata.basic.abstract = results["basic"]["abstract"]

            if results["basic"].get("title"):
                new_title = results["basic"]["title"]
                current_title = enriched.metadata.basic.title or ""
                if not current_title or len(new_title) > len(current_title):
                    enriched.metadata.basic.title = new_title

            if results["basic"].get("authors") and not enriched.metadata.basic.authors:
                enriched.metadata.basic.authors = results["basic"]["authors"]

            if results["basic"].get("year") and not enriched.metadata.basic.year:
                enriched.metadata.basic.year = results["basic"]["year"]

            if (
                results["basic"].get("keywords")
                and not enriched.metadata.basic.keywords
            ):
                enriched.metadata.basic.keywords = results["basic"]["keywords"]

        # Publication metadata
        if "publication" in results:
            pub = results["publication"]
            meta_pub = enriched.metadata.publication
            if pub.get("journal") and not meta_pub.journal:
                meta_pub.journal = pub["journal"]
            if pub.get("publisher") and not meta_pub.publisher:
                meta_pub.publisher = pub["publisher"]
            if pub.get("volume") and not meta_pub.volume:
                meta_pub.volume = pub["volume"]
            if pub.get("issue") and not meta_pub.issue:
                meta_pub.issue = pub["issue"]
            if pub.get("pages") and not meta_pub.pages:
                meta_pub.pages = pub["pages"]

        # Citation metadata
        if "citation_count" in results:
            count = results["citation_count"].get("count") or results[
                "citation_count"
            ].get("total")
            if count:
                current_count = enriched.metadata.citation_count.total or 0
                if not current_count or count > current_count:
                    enriched.metadata.citation_count.total = count

        # URL metadata
        if "url" in results:
            if results["url"].get("pdf"):
                pdf_url = results["url"]["pdf"]
                if not any(p.get("url") == pdf_url for p in enriched.metadata.url.pdfs):
                    enriched.metadata.url.pdfs.append(
                        {"url": pdf_url, "source": "enrichment"}
                    )
            if results["url"].get("url") and not enriched.metadata.url.publisher:
                enriched.metadata.url.publisher = results["url"]["url"]

        return enriched

    def _enrich_current_project(self) -> Dict[str, int]:
        """Enrich all papers in the current project.

        Returns
        -------
            Dictionary with enrichment statistics
        """
        if not self.project:
            raise ValueError(
                "No project specified. Use Scholar(project='name') "
                "or provide papers to enrich()."
            )

        papers = self.load_project(self.project)
        logger.info(
            f"{self.name}: Enriching {len(papers)} papers in project '{self.project}'"
        )

        enriched_papers = self.enrich_papers(papers)

        enriched_count = sum(
            1
            for i, p in enumerate(enriched_papers)
            if p.abstract and not papers[i].abstract
        )

        saved_ids = self.save_papers_to_library(enriched_papers)

        return {
            "enriched": enriched_count,
            "failed": len(papers) - enriched_count,
            "total": len(papers),
            "saved": len(saved_ids),
        }


# EOF
