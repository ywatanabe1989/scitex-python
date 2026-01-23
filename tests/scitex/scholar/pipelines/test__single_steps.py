# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/pipelines/_single_steps.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2026-01-14 (ywatanabe)"
# # File: src/scitex/scholar/pipelines/_single_steps.py
# """Step implementations for ScholarPipelineSingle."""
# 
# from __future__ import annotations
# 
# import hashlib
# from pathlib import Path
# from typing import TYPE_CHECKING
# 
# from scitex import logging
# from scitex.scholar.core import Paper
# 
# if TYPE_CHECKING:
#     from scitex.scholar.storage import PaperIO
# 
# logger = logging.getLogger(__name__)
# 
# 
# class PipelineStepsMixin:
#     """Mixin containing step implementations for single paper pipeline."""
# 
#     # ----------------------------------------
#     # Steps
#     # ----------------------------------------
#     def _step_01_normalize_as_doi(self, doi_or_title):
#         logger.info(f"{self.name}: Processing Query: {doi_or_title}")
#         is_doi = doi_or_title.strip().startswith("10.")
#         return doi_or_title.strip() if is_doi else None
# 
#     async def _step_02_create_paper(self, doi, doi_or_title):
#         """Create Paper object and resolve DOI from title if needed."""
#         paper = Paper()
#         if doi:
#             paper.metadata.id.doi = doi
#             paper.metadata.id.doi_engines = ["user_input"]
#         else:
#             from scitex.scholar.metadata_engines import ScholarEngine
# 
#             engine = ScholarEngine()
#             metadata_dict = await engine.search_async(title=doi_or_title)
#             if metadata_dict and metadata_dict.get("id", {}).get("doi"):
#                 doi = metadata_dict["id"]["doi"]
#                 paper.metadata.id.doi = doi
#                 paper.metadata.id.doi_engines = metadata_dict["id"].get(
#                     "doi_engines", ["ScholarEngine"]
#                 )
#                 logger.success(f"{self.name}: DOI resolved from title: {doi}")
#                 self._merge_metadata_into_paper(paper, metadata_dict)
#             else:
#                 logger.error(f"{self.name}: Could not resolve DOI: {doi_or_title}")
#                 raise ValueError(f"No DOI found for title: {doi_or_title}")
#         return paper
# 
#     def _step_03_add_paper_id(self, paper):
#         paper_id = self._generate_paper_id(paper.metadata.id.doi)
#         paper.container.library_id = paper_id
#         logger.info(f"{self.name}: Library ID: {paper_id}")
#         return paper
# 
#     async def _step_04_resolve_metadata(self, paper, io, force):
#         if not io.has_metadata() or force:
#             logger.info(f"{self.name}: Resolving metadata...")
#             from scitex.scholar.metadata_engines import ScholarEngine
# 
#             engine = ScholarEngine()
#             metadata_dict = await engine.search_async(doi=paper.metadata.id.doi)
#             if metadata_dict:
#                 self._merge_metadata_into_paper(paper, metadata_dict)
#                 self._enrich_impact_factor(paper)
#                 io.save_metadata()
#                 logger.success(f"{self.name}: Metadata enriched")
#             else:
#                 paper.metadata.basic.title = "Pending metadata resolution"
#                 paper.metadata.basic.title_engines = ["pending"]
#                 io.save_metadata()
#                 logger.warning(f"{self.name}: No metadata found")
#         else:
#             logger.info(f"{self.name}: Metadata exists, loading...")
#             paper = io.load_metadata()
#             if paper.metadata.basic.title == "Pending metadata resolution":
#                 logger.info(f"{self.name}: Enriching existing metadata...")
#                 from scitex.scholar.metadata_engines import ScholarEngine
# 
#                 engine = ScholarEngine()
#                 metadata_dict = await engine.search_async(doi=paper.metadata.id.doi)
#                 if metadata_dict:
#                     self._merge_metadata_into_paper(paper, metadata_dict)
#                     self._enrich_impact_factor(paper)
#                     io.save_metadata()
#                     logger.success(f"{self.name}: Metadata enriched")
#         return paper
# 
#     async def _step_05_setup_browser(self, paper, io):
#         needs_browser = not paper.metadata.url.pdfs or not io.has_pdf()
#         if not needs_browser:
#             return None, None, None
#         from scitex.scholar import ScholarAuthManager, ScholarBrowserManager
#         from scitex.scholar.auth import AuthenticationGateway
# 
#         logger.info(f"{self.name}: Setting up browser ({self.chrome_profile})...")
#         auth_manager = ScholarAuthManager()
#         browser_manager = ScholarBrowserManager(
#             chrome_profile_name=self.chrome_profile,
#             browser_mode=self.browser_mode,
#             auth_manager=auth_manager,
#         )
#         (
#             browser,
#             context,
#         ) = await browser_manager.get_authenticated_browser_and_context_async()
#         auth_gateway = AuthenticationGateway(
#             auth_manager=auth_manager, browser_manager=browser_manager
#         )
#         return browser_manager, context, auth_gateway
# 
#     async def _step_06_find_pdf_urls(self, paper, io, context, auth_gateway, force):
#         if not paper.metadata.url.pdfs or force:
#             logger.info(f"{self.name}: Finding PDF URLs...")
#             try:
#                 url_context = await auth_gateway.prepare_context_async(
#                     doi=paper.metadata.id.doi, context=context
#                 )
#                 publisher_url = (
#                     url_context.url if url_context else paper.metadata.id.doi
#                 )
#             except Exception as e:
#                 logger.warning(f"{self.name}: Auth gateway failed: {e}")
#                 publisher_url = paper.metadata.id.doi
#             from scitex.scholar import ScholarURLFinder
# 
#             url_finder = ScholarURLFinder(context)
#             urls = await url_finder.find_pdf_urls(publisher_url)
#             paper.metadata.url.pdfs = urls
#             paper.metadata.url.pdfs_engines = ["ScholarURLFinder"]
#             io.save_metadata()
#             logger.info(f"{self.name}: Found {len(urls)} PDF URL(s)")
#         else:
#             logger.info(f"{self.name}: PDF URLs exist ({len(paper.metadata.url.pdfs)})")
# 
#     async def _step_07_download_pdf(self, paper, io, context, auth_gateway, force):
#         if (not io.has_pdf() or force) and paper.metadata.url.pdfs:
#             logger.info(f"{self.name}: Downloading PDF...")
#             from scitex.scholar.pdf_download import ScholarPDFDownloader
# 
#             downloader = ScholarPDFDownloader(context)
#             downloaded, temp_path = await self._download_pdf_from_url(
#                 paper, io, context, auth_gateway, downloader
#             )
#             if downloaded:
#                 self._handle_downloaded_pdf(paper, io, downloaded, temp_path)
#             else:
#                 self._check_manual_download(io)
#         elif io.has_pdf():
#             logger.info(f"{self.name}: PDF already exists, skipping download")
# 
#     def _step_08_extract_content(self, io, force):
#         if io.has_pdf() and (not io.has_content() or force):
#             logger.info(f"{self.name}: Extracting content (text, tables, images)...")
#             import scitex
# 
#             try:
#                 pdf_path = io.get_pdf_path()
#                 content = scitex.io.load(
#                     str(pdf_path),
#                     ext="pdf",
#                     mode="scientific",
#                     output_dir=str(io.get_images_dir()),
#                 )
#                 if hasattr(content, "text") and content.text:
#                     io.save_text(content.text)
#                 if hasattr(content, "tables") and content.tables:
#                     io.save_tables_from_extraction(content.tables)
#                 stats = getattr(content, "stats", {})
#                 logger.info(
#                     f"{self.name}: Extracted {stats.get('num_tables', 0)} tables, "
#                     f"{stats.get('num_images', 0)} images"
#                 )
#             except Exception as e:
#                 logger.warning(f"{self.name}: Content extraction failed: {e}")
# 
#     def _step_09_link_to_project(self, paper, io, project):
#         if project:
#             logger.info(f"{self.name}: Linking to project: {project}")
#             return self._link_to_project(paper, project, io)
#         return None
# 
#     def _step_10_log_final_status(self, io):
#         logger.success(f"{self.name}: Complete")
#         for filename, exists in io.get_all_files().items():
#             logger.debug(f"  {'✓' if exists else '✗'} {filename}")
# 
#     # ----------------------------------------
#     # Step 07 Helpers
#     # ----------------------------------------
#     async def _download_pdf_from_url(
#         self, paper, io, context, auth_gateway, downloader
#     ):
#         pdf_url = paper.metadata.url.pdfs[0]
#         if isinstance(pdf_url, dict):
#             pdf_url = pdf_url["url"]
#         logger.info(f"{self.name}: PDF URL: {pdf_url}")
#         try:
#             await auth_gateway.prepare_context_async(
#                 doi=paper.metadata.id.doi, context=context
#             )
#         except Exception as e:
#             logger.warn(str(e))
#         temp_pdf_path = io.paper_dir / "temp.pdf"
#         downloaded_file = await downloader.download_from_url(
#             pdf_url, output_path=temp_pdf_path, doi=paper.metadata.id.doi
#         )
#         return downloaded_file, temp_pdf_path
# 
#     def _handle_downloaded_pdf(self, paper, io, downloaded_file, temp_pdf_path):
#         import shutil
# 
#         if downloaded_file == temp_pdf_path and temp_pdf_path.exists():
#             main_pdf = io.get_pdf_path()
#             shutil.move(str(temp_pdf_path), str(main_pdf))
#             paper.metadata.path.pdfs = [str(main_pdf)]
#             paper.container.pdf_size_bytes = main_pdf.stat().st_size
#             io.save_metadata()
#             logger.success(f"{self.name}: PDF downloaded to MASTER")
#         else:
#             io.save_pdf(downloaded_file)
#             io.save_metadata()
#         logger.info(f"{self.name}: PDF saved ({str(downloaded_file)})")
# 
#     def _check_manual_download(self, io):
#         import time
# 
#         from scitex.scholar import ScholarConfig
# 
#         logger.warning(f"{self.name}: Automated download returned None")
#         config = ScholarConfig()
#         downloads_dir = config.get_library_downloads_dir()
#         current_time = time.time()
#         recent_pdfs = [
#             (p, current_time - p.stat().st_mtime)
#             for p in downloads_dir.glob("*")
#             if p.is_file()
#             and p.stat().st_size > 100_000
#             and (current_time - p.stat().st_mtime) < 600
#         ]
#         if recent_pdfs:
#             recent_pdfs.sort(key=lambda x: x[1])
#             latest_pdf = recent_pdfs[0][0]
#             logger.info(f"{self.name}: Found recent PDF: {latest_pdf.name}")
#             io.save_pdf(latest_pdf)
#             io.save_metadata()
#             logger.success(f"{self.name}: Manual PDF saved to MASTER")
#         else:
#             logger.warning(f"{self.name}: No recent PDFs found")
# 
# 
# class PipelineHelpersMixin:
#     """Mixin containing helper methods for single paper pipeline."""
# 
#     def _generate_paper_id(self, doi: str) -> str:
#         """Generate 8-digit library ID from DOI."""
#         return hashlib.md5(f"DOI:{doi}".encode()).hexdigest()[:8].upper()
# 
#     def _link_to_project(self, paper: Paper, project: str, io: PaperIO) -> Path:
#         """Create human-readable symlink in project directory."""
#         from scitex.scholar import ScholarConfig
# 
#         config = ScholarConfig()
#         project_dir = config.path_manager.get_library_project_dir(project)
#         pdf_files = list(io.paper_dir.glob("*.pdf"))
#         entry_name = config.path_manager.get_library_project_entry_dirname(
#             n_pdfs=len(pdf_files),
#             citation_count=paper.metadata.citation_count.total or 0,
#             impact_factor=int(paper.metadata.publication.impact_factor or 0),
#             year=paper.metadata.basic.year or 0,
#             first_author=(
#                 paper.metadata.basic.authors[0].split()[-1]
#                 if paper.metadata.basic.authors
#                 else "Unknown"
#             ),
#             journal_name=(
#                 paper.metadata.publication.short_journal
#                 or paper.metadata.publication.journal
#                 or "Unknown"
#             ),
#         )
#         symlink_path = project_dir / entry_name
#         target_path = Path("../MASTER") / paper.container.library_id
#         if symlink_path.exists() or symlink_path.is_symlink():
#             symlink_path.unlink()
#         symlink_path.symlink_to(target_path)
#         logger.success(f"{self.name}: Created symlink: {project}/{entry_name}")
#         return symlink_path
# 
#     def _enrich_impact_factor(self, paper: Paper) -> None:
#         """Add journal impact factor to paper metadata if not present."""
#         if paper.metadata.publication.impact_factor:
#             return
#         journal = (
#             paper.metadata.publication.short_journal
#             or paper.metadata.publication.journal
#         )
#         if not journal:
#             return
#         try:
#             from scitex.scholar.impact_factor import ImpactFactorEngine
# 
#             if_engine = ImpactFactorEngine()
#             metrics = if_engine.get_metrics(journal)
#             if metrics and metrics.get("impact_factor"):
#                 paper.metadata.publication.impact_factor = metrics["impact_factor"]
#                 paper.metadata.publication.impact_factor_engines = [
#                     metrics.get("source", "JCR")
#                 ]
#                 logger.info(f"{self.name}: IF added: {metrics['impact_factor']}")
#         except Exception as e:
#             logger.debug(f"{self.name}: IF lookup failed: {e}")
# 
#     def _merge_metadata_into_paper(self, paper: Paper, metadata_dict: dict) -> None:
#         """Merge metadata dictionary from ScholarEngine into Paper object."""
# 
#         def update_field(section, field_name, value, engines):
#             if value is None:
#                 return
#             # Type conversions
#             if section == "id" and not isinstance(value, str):
#                 value = str(value)
#             if field_name == "year" and not isinstance(value, int):
#                 try:
#                     value = int(value)
#                 except (ValueError, TypeError):
#                     return
#             if section == "citation_count" and not isinstance(value, int):
#                 try:
#                     value = int(value)
#                 except (ValueError, TypeError):
#                     return
#             try:
#                 section_obj = getattr(paper.metadata, section)
#                 setattr(section_obj, field_name, value)
#                 setattr(section_obj, f"{field_name}_engines", engines)
#             except Exception:
#                 pass
# 
#         # ID section
#         if "id" in metadata_dict:
#             id_data = metadata_dict["id"]
#             for field in [
#                 "doi",
#                 "arxiv_id",
#                 "pmid",
#                 "corpus_id",
#                 "semantic_id",
#                 "ieee_id",
#                 "scholar_id",
#             ]:
#                 if field in id_data:
#                     update_field(
#                         "id", field, id_data[field], id_data.get(f"{field}_engines", [])
#                     )
# 
#         # Basic section
#         if "basic" in metadata_dict:
#             basic_data = metadata_dict["basic"]
#             for field in ["title", "authors", "year", "abstract", "keywords", "type"]:
#                 if field in basic_data:
#                     update_field(
#                         "basic",
#                         field,
#                         basic_data[field],
#                         basic_data.get(f"{field}_engines", []),
#                     )
# 
#         # Citation count section
#         if "citation_count" in metadata_dict:
#             cc_data = metadata_dict["citation_count"]
#             if "total" in cc_data:
#                 update_field(
#                     "citation_count",
#                     "total",
#                     cc_data["total"],
#                     cc_data.get("total_engines", []),
#                 )
#             for year in range(2015, 2026):
#                 if str(year) in cc_data:
#                     update_field(
#                         "citation_count",
#                         f"y{year}",
#                         cc_data[str(year)],
#                         cc_data.get(f"{year}_engines", []),
#                     )
# 
#         # Publication section
#         if "publication" in metadata_dict:
#             pub_data = metadata_dict["publication"]
#             for field in [
#                 "journal",
#                 "short_journal",
#                 "impact_factor",
#                 "issn",
#                 "volume",
#                 "issue",
#                 "first_page",
#                 "last_page",
#                 "pages",
#                 "publisher",
#             ]:
#                 if field in pub_data:
#                     update_field(
#                         "publication",
#                         field,
#                         pub_data[field],
#                         pub_data.get(f"{field}_engines", []),
#                     )
# 
#         # URL section
#         if "url" in metadata_dict:
#             url_data = metadata_dict["url"]
#             for field in ["doi", "publisher", "arxiv", "corpus_id"]:
#                 if field in url_data:
#                     update_field(
#                         "url",
#                         field,
#                         url_data[field],
#                         url_data.get(f"{field}_engines", []),
#                     )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/pipelines/_single_steps.py
# --------------------------------------------------------------------------------
