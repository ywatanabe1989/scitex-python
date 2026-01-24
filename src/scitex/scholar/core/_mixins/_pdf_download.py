#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/_pdf_download.py

"""
PDF download mixin for Scholar class.

Provides PDF downloading functionality from DOIs and BibTeX files.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from scitex import logging
from scitex.scholar.auth.core.AuthenticationGateway import AuthenticationGateway
from scitex.scholar.pdf_download.ScholarPDFDownloader import ScholarPDFDownloader

if TYPE_CHECKING:
    from ..Papers import Papers

logger = logging.getLogger(__name__)


class PDFDownloadMixin:
    """Mixin providing PDF download methods."""

    async def download_pdfs_from_dois_async(
        self,
        dois: List[str],
        output_dir: Optional[Path] = None,
        max_concurrent: int = 1,
    ) -> Dict[str, int]:
        """Download PDFs for given DOIs using ScholarPDFDownloader.

        Args:
            dois: List of DOI strings
            output_dir: Output directory (not used - downloads to library MASTER)
            max_concurrent: Maximum concurrent downloads (default: 1 for sequential)

        Returns
        -------
            Dictionary with download statistics
        """
        if not dois:
            return {"downloaded": 0, "failed": 0, "errors": 0}

        (
            browser,
            context,
        ) = await self._browser_manager.get_authenticated_browser_and_context_async()

        try:
            pdf_downloader = ScholarPDFDownloader(
                context=context,
                config=self.config,
            )

            logger.info(
                f"{self.name}: Starting PDF download for {len(dois)} DOIs "
                f"(max_concurrent={max_concurrent})"
            )

            results = await pdf_downloader.download_from_dois(
                dois=dois,
                output_dir=str(output_dir) if output_dir else "/tmp/",
                max_concurrent=max_concurrent,
            )

            stats = {"downloaded": 0, "failed": 0, "errors": 0}
            library_dir = self.config.path_manager.library_dir
            master_dir = library_dir / "MASTER"
            master_dir.mkdir(parents=True, exist_ok=True)

            for doi, downloaded_paths in zip(dois, results):
                try:
                    if downloaded_paths and len(downloaded_paths) > 0:
                        temp_pdf_path = downloaded_paths[0]

                        paper_id = self.config.path_manager._generate_paper_id(doi=doi)
                        storage_path = master_dir / paper_id
                        storage_path.mkdir(parents=True, exist_ok=True)

                        pdf_filename = (
                            f"DOI_{doi.replace('/', '_').replace(':', '_')}.pdf"
                        )
                        master_pdf_path = storage_path / pdf_filename
                        shutil.move(str(temp_pdf_path), str(master_pdf_path))

                        metadata_file = storage_path / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                        else:
                            metadata = {
                                "doi": doi,
                                "scitex_id": paper_id,
                                "created_at": datetime.now().isoformat(),
                                "created_by": "SciTeX Scholar",
                            }

                        metadata["pdf_path"] = str(
                            master_pdf_path.relative_to(library_dir)
                        )
                        metadata["pdf_downloaded_at"] = datetime.now().isoformat()
                        metadata["pdf_size_bytes"] = master_pdf_path.stat().st_size
                        metadata["updated_at"] = datetime.now().isoformat()

                        with open(metadata_file, "w") as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)

                        if self.project not in ["master", "MASTER"]:
                            self._library_manager.update_symlink(
                                master_storage_path=storage_path,
                                project=self.project,
                            )

                        logger.success(
                            f"{self.name}: Downloaded and organized PDF for {doi}: "
                            f"{master_pdf_path}"
                        )
                        stats["downloaded"] += 1
                    else:
                        logger.warning(f"{self.name}: No PDF downloaded for DOI: {doi}")
                        stats["failed"] += 1

                except Exception as e:
                    logger.error(f"{self.name}: Failed to organize PDF for {doi}: {e}")
                    stats["errors"] += 1
                    stats["failed"] += 1

            return stats

        finally:
            await self._browser_manager.close()

    async def _download_pdfs_sequential(
        self, dois: List[str], output_dir: Optional[Path] = None
    ) -> Dict[str, int]:
        """Sequential PDF download with authentication gateway."""
        results = {"downloaded": 0, "failed": 0, "errors": 0}

        (
            browser,
            context,
        ) = await self._browser_manager.get_authenticated_browser_and_context_async()

        auth_gateway = AuthenticationGateway(
            auth_manager=self._auth_manager,
            browser_manager=self._browser_manager,
            config=self.config,
        )

        pdf_downloader = ScholarPDFDownloader(
            context=context,
            config=self.config,
        )

        library_dir = self.config.path_manager.library_dir
        master_dir = library_dir / "MASTER"
        project_dir = library_dir / self.project
        master_dir.mkdir(parents=True, exist_ok=True)
        project_dir.mkdir(parents=True, exist_ok=True)

        for doi in dois:
            try:
                logger.info(f"{self.name}: Processing DOI: {doi}")

                _url_context = await auth_gateway.prepare_context_async(
                    doi=doi, context=context
                )

                urls = await self._find_urls_for_doi_async(doi, context)
                pdf_urls = urls.get("urls_pdf", [])

                if not pdf_urls:
                    logger.warning(f"{self.name}: No PDF URLs found for DOI: {doi}")
                    results["failed"] += 1
                    continue

                downloaded_path = None
                for pdf_entry in pdf_urls:
                    pdf_url = (
                        pdf_entry.get("url")
                        if isinstance(pdf_entry, dict)
                        else pdf_entry
                    )

                    if not pdf_url:
                        continue

                    temp_output = (
                        Path("/tmp") / f"{doi.replace('/', '_').replace(':', '_')}.pdf"
                    )

                    result = await pdf_downloader.download_from_url(
                        pdf_url=pdf_url, output_path=temp_output
                    )

                    if result and result.exists():
                        downloaded_path = result
                        break

                if downloaded_path:
                    self._store_downloaded_pdf(
                        doi, downloaded_path, library_dir, master_dir
                    )
                    downloaded_path.unlink()
                    results["downloaded"] += 1
                else:
                    logger.warning(
                        f"{self.name}: Failed to download any PDF for DOI: {doi}"
                    )
                    results["failed"] += 1

            except Exception as e:
                logger.error(f"{self.name}: Failed to process {doi}: {e}")
                results["errors"] += 1
                results["failed"] += 1

        await self._browser_manager.close()
        logger.info(f"{self.name}: PDF download complete: {results}")
        return results

    def _store_downloaded_pdf(
        self,
        doi: str,
        downloaded_path: Path,
        library_dir: Path,
        master_dir: Path,
    ) -> None:
        """Store downloaded PDF in library structure."""
        from ..Paper import Paper
        from ..Papers import Papers

        paper_id = self.config.path_manager._generate_paper_id(doi=doi)
        storage_path = master_dir / paper_id
        storage_path.mkdir(parents=True, exist_ok=True)

        readable_name = None
        temp_paper = None
        try:
            temp_paper = Paper()
            temp_paper.metadata.id.doi = doi
            temp_papers = Papers([temp_paper])
            enriched = asyncio.run(self.enrich_papers_async(temp_papers))
            if enriched and len(enriched) > 0:
                temp_paper = enriched[0]

            first_author = "Unknown"
            authors = temp_paper.metadata.basic.authors
            if authors and len(authors) > 0:
                author_parts = authors[0].split()
                if len(author_parts) > 1:
                    first_author = author_parts[-1]
                else:
                    first_author = author_parts[0]

            year = temp_paper.metadata.basic.year
            year_str = str(year) if year else "Unknown"

            journal_clean = "Unknown"
            journal = temp_paper.metadata.publication.journal
            if journal:
                journal_clean = "".join(
                    c for c in journal if c.isalnum() or c in " "
                ).replace(" ", "")
                if not journal_clean:
                    journal_clean = "Unknown"

            readable_name = f"{first_author}-{year_str}-{journal_clean}"
        except Exception:
            pass

        if not readable_name:
            readable_name = f"DOI_{doi.replace('/', '_').replace(':', '_')}"

        pdf_filename = f"DOI_{doi.replace('/', '_').replace(':', '_')}.pdf"
        master_pdf_path = storage_path / pdf_filename
        shutil.copy2(downloaded_path, master_pdf_path)

        metadata_file = storage_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            logger.debug(f"{self.name}: Loaded existing metadata for {paper_id}")
        else:
            metadata = {
                "doi": doi,
                "scitex_id": paper_id,
                "created_at": datetime.now().isoformat(),
                "created_by": "SciTeX Scholar",
            }

            if temp_paper:
                paper_dict = temp_paper.to_dict()
                for key, value in paper_dict.items():
                    if value is not None and key not in ["doi", "scitex_id"]:
                        metadata[key] = value

        metadata["pdf_path"] = str(master_pdf_path.relative_to(library_dir))
        metadata["pdf_downloaded_at"] = datetime.now().isoformat()
        metadata["pdf_size_bytes"] = master_pdf_path.stat().st_size
        metadata["updated_at"] = datetime.now().isoformat()

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if self.project not in ["master", "MASTER"]:
            self._library_manager.update_symlink(
                master_storage_path=storage_path,
                project=self.project,
            )

        logger.success(
            f"{self.name}: Downloaded PDF for {doi}: MASTER/{paper_id}/{pdf_filename}"
        )

    def download_pdfs_from_dois(
        self, dois: List[str], output_dir: Optional[Path] = None
    ) -> Dict[str, int]:
        """Download PDFs for given DOIs.

        Args:
            dois: List of DOI strings
            output_dir: Output directory (uses config default if None)

        Returns
        -------
            Dictionary with download statistics
        """
        return asyncio.run(self.download_pdfs_from_dois_async(dois, output_dir))

    def download_pdfs_from_bibtex(
        self,
        bibtex_input: Union[str, Path, Papers],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, int]:
        """Download PDFs from BibTeX file or Papers collection.

        Args:
            bibtex_input: BibTeX file path, content string, or Papers collection
            output_dir: Output directory (uses config default if None)

        Returns
        -------
            Dictionary with download statistics
        """
        from ..Papers import Papers

        if isinstance(bibtex_input, Papers):
            papers = bibtex_input
        else:
            papers = self.load_bibtex(bibtex_input)

        dois = [paper.metadata.id.doi for paper in papers if paper.metadata.id.doi]

        if not dois:
            logger.warning(f"{self.name}: No papers with DOIs found in BibTeX input")
            return {"downloaded": 0, "failed": 0, "errors": 0}

        logger.info(
            f"{self.name}: Found {len(dois)} papers with DOIs "
            f"out of {len(papers)} total papers"
        )

        return self.download_pdfs_from_dois(dois, output_dir)


# EOF
