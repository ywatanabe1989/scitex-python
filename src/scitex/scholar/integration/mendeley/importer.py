#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mendeley importer - Import from Mendeley to Scholar library.

Uses mendeley python SDK: https://github.com/Mendeley/mendeley-python-sdk
Install: pip install mendeley
"""

from typing import List, Optional
import os

from scitex import logging
from scitex.scholar.core.Papers import Papers
from ..base import BaseImporter
from .mapper import MendeleyMapper

logger = logging.getLogger(__name__)


class MendeleyImporter(BaseImporter):
    """Import from Mendeley to Scholar library."""

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        project: str = "default",
        config=None,
    ):
        """Initialize Mendeley importer.

        Args:
            app_id: Mendeley app ID
            app_secret: Mendeley app secret
            access_token: OAuth access token
            project: Scholar project name
            config: Optional ScholarConfig instance

        Get credentials from: https://dev.mendeley.com/myapps.html
        """
        credentials = {
            "app_id": app_id or os.getenv("MENDELEY_APP_ID"),
            "app_secret": app_secret or os.getenv("MENDELEY_APP_SECRET"),
            "access_token": access_token or os.getenv("MENDELEY_ACCESS_TOKEN"),
        }

        super().__init__(credentials=credentials, project=project, config=config)

        self._client = None

    def _create_mapper(self) -> MendeleyMapper:
        """Create Mendeley mapper."""
        return MendeleyMapper(config=self.config)

    def _get_client(self):
        """Get or create Mendeley client."""
        if self._client is None:
            try:
                from mendeley import Mendeley

                mendeley = Mendeley(
                    client_id=self.credentials["app_id"],
                    client_secret=self.credentials["app_secret"],
                )

                if self.credentials.get("access_token"):
                    # Use existing access token
                    session = mendeley.start_client_credentials_flow().authenticate()
                    self._client = session
                else:
                    # Start OAuth flow
                    auth = mendeley.start_authorization_code_flow()
                    logger.info(f"Please authorize at: {auth.get_login_url()}")
                    logger.info("Then provide the authorization code")
                    # This requires interactive input - implement as needed

                logger.info("Connected to Mendeley")

            except ImportError:
                raise ImportError(
                    "mendeley SDK not installed. Install with: pip install mendeley"
                )

        return self._client

    def import_collection(
        self,
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        limit: Optional[int] = None,
        include_pdfs: bool = True,
        include_annotations: bool = True,
    ) -> Papers:
        """Import items from a Mendeley folder.

        Args:
            collection_id: Folder ID
            collection_name: Folder name
            limit: Maximum items to import
            include_pdfs: Include PDF files
            include_annotations: Include annotations

        Returns:
            Papers collection
        """
        client = self._get_client()

        # Get folder if specified
        folder = None
        if collection_name:
            folders = client.folders.list()
            for f in folders.items:
                if f.name == collection_name:
                    folder = f
                    collection_id = f.id
                    break

        # Get documents
        if folder:
            logger.info(f"Importing from folder: {folder.name}")
            documents = folder.documents.list(limit=limit).items
        else:
            logger.info("Importing all documents")
            documents = client.documents.list(limit=limit).items

        logger.info(f"Retrieved {len(documents)} documents from Mendeley")

        # Convert to Papers
        papers = []
        for i, doc in enumerate(documents, 1):
            try:
                doc_dict = doc.json
                paper = self.mapper.external_to_paper(doc_dict)

                # Import files if requested
                if include_pdfs:
                    files = doc.files.list().items
                    for file in files:
                        if file.mime_type == "application/pdf":
                            pdf_entry = {
                                "url": file.file_url,
                                "source": "mendeley",
                                "mendeley_file_id": file.id,
                            }
                            paper.metadata.url.pdfs.append(pdf_entry)

                papers.append(paper)

                logger.info(
                    f"[{i}/{len(documents)}] Imported: {paper.metadata.basic.title[:50]}..."
                )

            except Exception as e:
                logger.error(f"Failed to import document: {e}")
                continue

        papers_collection = Papers(papers, project=self.project)
        logger.success(f"Imported {len(papers)} papers from Mendeley")

        return papers_collection

    def import_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: Optional[int] = None,
        include_pdfs: bool = True,
        include_annotations: bool = True,
    ) -> Papers:
        """Import items with specific tags.

        Args:
            tags: List of tag names
            match_all: Require all tags
            limit: Maximum items
            include_pdfs: Include PDFs
            include_annotations: Include annotations

        Returns:
            Papers collection
        """
        client = self._get_client()

        # Search by tags
        all_docs = client.documents.list(limit=limit).items

        matching_docs = []
        for doc in all_docs:
            doc_tags = doc.json.get("tags", [])

            if match_all:
                if all(tag in doc_tags for tag in tags):
                    matching_docs.append(doc)
            else:
                if any(tag in doc_tags for tag in tags):
                    matching_docs.append(doc)

        logger.info(f"Found {len(matching_docs)} documents matching tags: {tags}")

        # Convert to Papers
        papers = []
        for i, doc in enumerate(matching_docs, 1):
            try:
                doc_dict = doc.json
                paper = self.mapper.external_to_paper(doc_dict)

                if include_pdfs:
                    files = doc.files.list().items
                    for file in files:
                        if file.mime_type == "application/pdf":
                            pdf_entry = {
                                "url": file.file_url,
                                "source": "mendeley",
                                "mendeley_file_id": file.id,
                            }
                            paper.metadata.url.pdfs.append(pdf_entry)

                papers.append(paper)

                logger.info(
                    f"[{i}/{len(matching_docs)}] Imported: {paper.metadata.basic.title[:50]}..."
                )

            except Exception as e:
                logger.error(f"Failed to import document: {e}")
                continue

        papers_collection = Papers(papers, project=self.project)
        logger.success(f"Imported {len(papers)} papers by tags")

        return papers_collection


# EOF
