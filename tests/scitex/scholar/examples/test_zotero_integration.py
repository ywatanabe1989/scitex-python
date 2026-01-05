# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/examples/zotero_integration.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Zotero Integration Examples for SciTeX Scholar.
# 
# Demonstrates:
# - Importing from Zotero
# - Exporting to Zotero
# - Live synchronization
# - Citation insertion
# """
# 
# import os
# from pathlib import Path
# 
# from scitex import logging
# from scitex.scholar.zotero import ZoteroImporter, ZoteroExporter, ZoteroLinker
# from scitex.scholar.core import Papers
# 
# logger = logging.getLogger(__name__)
# 
# 
# def example_import():
#     """Example: Import papers from Zotero."""
#     logger.info("=" * 80)
#     logger.info("Example: Import from Zotero")
#     logger.info("=" * 80)
# 
#     # Get credentials from environment or use your own
#     library_id = os.getenv("ZOTERO_LIBRARY_ID", "YOUR_LIBRARY_ID")
#     api_key = os.getenv("ZOTERO_API_KEY", "YOUR_API_KEY")
# 
#     # Initialize importer
#     importer = ZoteroImporter(
#         library_id=library_id,
#         library_type="user",
#         api_key=api_key,
#         project="zotero_test",
#     )
# 
#     # Import specific collection
#     logger.info("\n1. Import from collection:")
#     papers = importer.import_collection(
#         collection_name="Machine Learning",  # Change to your collection name
#         limit=5,  # Limit for testing
#         include_pdfs=True,
#         include_annotations=True,
#     )
#     logger.success(f"Imported {len(papers)} papers from collection")
# 
#     # Import by tags
#     logger.info("\n2. Import by tags:")
#     papers = importer.import_by_tags(
#         tags=["deep learning", "transformers"],
#         match_all=False,  # OR logic
#         limit=3,
#     )
#     logger.success(f"Imported {len(papers)} papers by tags")
# 
#     # Save to Scholar library
#     logger.info("\n3. Save to Scholar library:")
#     results = importer.import_to_library(papers)
#     logger.success(f"Saved {len(results)} papers to Scholar library")
# 
#     for title, paper_id in list(results.items())[:3]:
#         logger.info(f"  - {title[:50]}... → {paper_id}")
# 
# 
# def example_export():
#     """Example: Export papers to Zotero."""
#     logger.info("\n" + "=" * 80)
#     logger.info("Example: Export to Zotero")
#     logger.info("=" * 80)
# 
#     library_id = os.getenv("ZOTERO_LIBRARY_ID", "YOUR_LIBRARY_ID")
#     api_key = os.getenv("ZOTERO_API_KEY", "YOUR_API_KEY")
# 
#     # Initialize exporter
#     exporter = ZoteroExporter(
#         library_id=library_id,
#         library_type="user",
#         api_key=api_key,
#         project="zotero_test",
#     )
# 
#     # Load papers from Scholar library
#     logger.info("\n1. Load papers from Scholar:")
#     papers = Papers.from_project("zotero_test")
#     logger.info(f"Loaded {len(papers)} papers")
# 
#     # Export to Zotero
#     logger.info("\n2. Export to Zotero collection:")
#     results = exporter.export_papers(
#         papers,
#         collection_name="From SciTeX Scholar",
#         create_collection=True,
#         update_existing=True,
#     )
#     logger.success(f"Exported {len(results)} papers to Zotero")
# 
#     # Export as BibTeX
#     logger.info("\n3. Export as BibTeX:")
#     output_dir = Path("./output")
#     output_dir.mkdir(exist_ok=True)
# 
#     bibtex_path = exporter.export_as_bibtex(
#         papers, output_path=output_dir / "papers.bib"
#     )
#     logger.success(f"Exported BibTeX: {bibtex_path}")
# 
#     # Export as RIS
#     logger.info("\n4. Export as RIS:")
#     ris_path = exporter.export_as_ris(papers, output_path=output_dir / "papers.ris")
#     logger.success(f"Exported RIS: {ris_path}")
# 
# 
# def example_sync():
#     """Example: Live synchronization with Zotero."""
#     logger.info("\n" + "=" * 80)
#     logger.info("Example: Live Synchronization")
#     logger.info("=" * 80)
# 
#     library_id = os.getenv("ZOTERO_LIBRARY_ID", "YOUR_LIBRARY_ID")
#     api_key = os.getenv("ZOTERO_API_KEY", "YOUR_API_KEY")
# 
#     # Initialize linker
#     linker = ZoteroLinker(
#         library_id=library_id,
#         library_type="user",
#         api_key=api_key,
#         project="zotero_test",
#         sync_interval=30,  # 30 seconds
#     )
# 
#     # Register callback
#     def on_sync_event(event_type, paper):
#         logger.info(f"[{event_type}] {paper.metadata.basic.title[:50]}...")
# 
#     linker.register_callback(on_sync_event)
# 
#     # Perform single sync
#     logger.info("\n1. Perform single sync:")
#     stats = linker.sync_once(bidirectional=True, auto_import=True, auto_export=False)
#     logger.success(f"Sync stats: {stats}")
# 
#     # Get sync status
#     logger.info("\n2. Get sync status:")
#     status = linker.get_sync_status()
#     logger.info(f"Status: {status}")
# 
#     # Note: For continuous sync, uncomment:
#     # logger.info("\n3. Start continuous sync (Ctrl+C to stop):")
#     # linker.start_sync(
#     #     bidirectional=True,
#     #     auto_import=True,
#     #     auto_export=False
#     # )
# 
# 
# def example_citations():
#     """Example: Citation insertion."""
#     logger.info("\n" + "=" * 80)
#     logger.info("Example: Citation Insertion")
#     logger.info("=" * 80)
# 
#     library_id = os.getenv("ZOTERO_LIBRARY_ID", "YOUR_LIBRARY_ID")
#     api_key = os.getenv("ZOTERO_API_KEY", "YOUR_API_KEY")
# 
#     # Initialize linker
#     linker = ZoteroLinker(
#         library_id=library_id,
#         library_type="user",
#         api_key=api_key,
#         project="zotero_test",
#     )
# 
#     # Load a sample paper
#     papers = Papers.from_project("zotero_test")
#     if not papers or len(papers) == 0:
#         logger.warning("No papers found for citation examples")
#         return
# 
#     paper = papers.papers[0]
#     logger.info(f"\nPaper: {paper.metadata.basic.title}")
# 
#     # BibTeX citation
#     logger.info("\n1. BibTeX format:")
#     bibtex = linker.insert_citation(paper, format="bibtex")
#     logger.info(f"\n{bibtex}")
# 
#     # RIS citation
#     logger.info("\n2. RIS format:")
#     ris = linker.insert_citation(paper, format="ris")
#     logger.info(f"\n{ris}")
# 
#     # Formatted text citations
#     logger.info("\n3. APA format:")
#     apa = linker.insert_citation(paper, format="text", style="apa")
#     logger.info(f"  {apa}")
# 
#     logger.info("\n4. MLA format:")
#     mla = linker.insert_citation(paper, format="text", style="mla")
#     logger.info(f"  {mla}")
# 
#     logger.info("\n5. Chicago format:")
#     chicago = linker.insert_citation(paper, format="text", style="chicago")
#     logger.info(f"  {chicago}")
# 
# 
# def main():
#     """Run all examples."""
#     logger.info("\n\n")
#     logger.info("*" * 80)
#     logger.info("SciTeX Scholar - Zotero Integration Examples")
#     logger.info("*" * 80)
# 
#     # Check credentials
#     if not os.getenv("ZOTERO_LIBRARY_ID") or not os.getenv("ZOTERO_API_KEY"):
#         logger.warning(
#             "\nPlease set environment variables:\n"
#             "  export ZOTERO_LIBRARY_ID=your_library_id\n"
#             "  export ZOTERO_API_KEY=your_api_key\n\n"
#             "Get your API key from: https://www.zotero.org/settings/keys\n"
#         )
#         logger.info("Running with placeholder values (examples will fail)")
# 
#     try:
#         # Run examples
#         example_import()
#         example_export()
#         example_sync()
#         example_citations()
# 
#     except Exception as e:
#         logger.error(f"Example failed: {e}")
#         import traceback
# 
#         traceback.print_exc()
# 
#     logger.info("\n" + "*" * 80)
#     logger.success("Examples complete!")
#     logger.info("*" * 80)
# 
# 
# if __name__ == "__main__":
#     main()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/examples/zotero_integration.py
# --------------------------------------------------------------------------------
