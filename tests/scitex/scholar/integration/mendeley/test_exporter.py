# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/integration/mendeley/exporter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Mendeley exporter - Export from Scholar library to Mendeley.
# """
# 
# from typing import Dict, List, Optional, Union
# import os
# 
# from scitex import logging
# from scitex.scholar.core.Paper import Paper
# from scitex.scholar.core.Papers import Papers
# from ..base import BaseExporter
# from .mapper import MendeleyMapper
# 
# logger = logging.getLogger(__name__)
# 
# 
# class MendeleyExporter(BaseExporter):
#     """Export from Scholar library to Mendeley."""
# 
#     def __init__(
#         self,
#         app_id: Optional[str] = None,
#         app_secret: Optional[str] = None,
#         access_token: Optional[str] = None,
#         project: str = "default",
#         config=None,
#     ):
#         """Initialize Mendeley exporter.
# 
#         Args:
#             app_id: Mendeley app ID
#             app_secret: Mendeley app secret
#             access_token: OAuth access token
#             project: Scholar project name
#             config: Optional ScholarConfig instance
#         """
#         credentials = {
#             "app_id": app_id or os.getenv("MENDELEY_APP_ID"),
#             "app_secret": app_secret or os.getenv("MENDELEY_APP_SECRET"),
#             "access_token": access_token or os.getenv("MENDELEY_ACCESS_TOKEN"),
#         }
# 
#         super().__init__(credentials=credentials, project=project, config=config)
# 
#         self._client = None
# 
#     def _create_mapper(self) -> MendeleyMapper:
#         """Create Mendeley mapper."""
#         return MendeleyMapper(config=self.config)
# 
#     def _get_client(self):
#         """Get or create Mendeley client."""
#         if self._client is None:
#             try:
#                 from mendeley import Mendeley
# 
#                 mendeley = Mendeley(
#                     client_id=self.credentials["app_id"],
#                     client_secret=self.credentials["app_secret"],
#                 )
# 
#                 session = mendeley.start_client_credentials_flow().authenticate()
#                 self._client = session
# 
#                 logger.info("Connected to Mendeley")
# 
#             except ImportError:
#                 raise ImportError(
#                     "mendeley SDK not installed. Install with: pip install mendeley"
#                 )
# 
#         return self._client
# 
#     def export_papers(
#         self,
#         papers: Union[Papers, List[Paper]],
#         collection_name: Optional[str] = None,
#         create_collection: bool = True,
#         update_existing: bool = True,
#     ) -> Dict[str, str]:
#         """Export Papers to Mendeley library.
# 
#         Args:
#             papers: Papers collection or list
#             collection_name: Target folder name
#             create_collection: Create folder if not exists
#             update_existing: Update existing documents
# 
#         Returns:
#             Dict mapping titles to Mendeley document IDs
#         """
#         client = self._get_client()
# 
#         if isinstance(papers, Papers):
#             paper_list = papers.papers
#         else:
#             paper_list = papers
# 
#         # Get or create folder
#         folder = None
#         if collection_name:
#             folders = client.folders.list()
#             for f in folders.items:
#                 if f.name == collection_name:
#                     folder = f
#                     break
# 
#             if not folder and create_collection:
#                 logger.info(f"Creating folder: {collection_name}")
#                 folder = client.folders.create(name=collection_name)
# 
#         results = {}
# 
#         for i, paper in enumerate(paper_list, 1):
#             try:
#                 # Convert to Mendeley format
#                 mendeley_doc = self.mapper.paper_to_external(paper)
# 
#                 # Check if document exists
#                 existing_doc = None
#                 if update_existing and paper.metadata.id.doi:
#                     # Search by DOI
#                     search_results = client.documents.search(
#                         f'doi:"{paper.metadata.id.doi}"'
#                     )
#                     if search_results.items:
#                         existing_doc = search_results.items[0]
# 
#                 if existing_doc:
#                     # Update existing
#                     logger.info(f"[{i}/{len(paper_list)}] Updating: {existing_doc.id}")
#                     # Mendeley SDK doesn't support updates directly
#                     # Would need to delete and recreate
#                     doc_id = existing_doc.id
#                 else:
#                     # Create new
#                     logger.info(
#                         f"[{i}/{len(paper_list)}] Creating: {paper.metadata.basic.title[:50]}..."
#                     )
#                     created_doc = client.documents.create(**mendeley_doc)
#                     doc_id = created_doc.id
# 
#                     # Add to folder if specified
#                     if folder:
#                         folder.add_document(created_doc)
# 
#                 results[paper.metadata.basic.title or f"paper_{i}"] = doc_id
# 
#                 logger.success(f"[{i}/{len(paper_list)}] Exported: {doc_id}")
# 
#             except Exception as e:
#                 logger.error(
#                     f"Failed to export paper: {paper.metadata.basic.title[:50]}... - {e}"
#                 )
#                 continue
# 
#         logger.success(f"Exported {len(results)}/{len(paper_list)} papers to Mendeley")
# 
#         return results
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/integration/mendeley/exporter.py
# --------------------------------------------------------------------------------
