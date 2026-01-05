# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/core/registry.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-10 01:11:16 (ywatanabe)"
# # File: /home/ywatanabe/proj/zotero-translators-python/src/zotero_translators_python/core/registry.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/zotero_translators_python/core/registry.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """Translator registry for managing and discovering translators.
# 
# The registry finds the right translator for a URL and extracts PDF links.
# 
# Usage:
#     from scitex.scholar.url_finder.translators import TranslatorRegistry
# 
#     # Find translator
#     translator = TranslatorRegistry.get_translator_for_url(url)
# 
#     # Extract PDFs
#     pdf_urls = await TranslatorRegistry.extract_pdf_urls_async(url, page)
# """
# 
# import logging
# from typing import List, Optional, Type
# 
# from playwright.async_api import Page
# 
# from .base import BaseTranslator
# 
# logger = logging.getLogger(__name__)
# 
# from ..individual.abc_news_australia import ABCNewsAustraliaTranslator
# from ..individual.access_engineering import AccessEngineeringTranslator
# from ..individual.access_science import AccessScienceTranslator
# from ..individual.acls_humanities_ebook import ACLSHumanitiesEBookTranslator
# from ..individual.aclweb import ACLWebTranslator
# from ..individual.acm import ACMTranslator
# from ..individual.acs import ACSTranslator
# 
# # from ..individual.cambridge_core import CambridgeCoreTranslator  # Not yet implemented
# from ..individual.acs_publications import ACSPublicationsTranslator
# from ..individual.adam_matthew_digital import AdamMatthewDigitalTranslator
# from ..individual.ads_bibcode import ADSBibcodeTranslator
# from ..individual.aea_web import AEAWebTranslator
# from ..individual.agris import AGRISTranslator
# from ..individual.aip import AIPTranslator
# from ..individual.all_africa import AllAfricaTranslator
# from ..individual.ams_journals import AMSJournalsTranslator
# from ..individual.annual_reviews import AnnualReviewsTranslator
# from ..individual.aosic import AOSICTranslator
# from ..individual.aps import APSTranslator
# from ..individual.aps_physics import APSPhysicsTranslator
# from ..individual.arxiv import ArXivTranslator
# from ..individual.arxiv_org import ArXivOrgTranslator
# from ..individual.asce import ASCETranslator
# 
# # from ..individual.atypon import AtyponTranslator  # Deprecated - use AtyponJournalsTranslator
# from ..individual.atypon_journals import AtyponJournalsTranslator
# from ..individual.biomed_central import BioMedCentralTranslator
# from ..individual.bioone import BioOneTranslator
# from ..individual.biorxiv import BioRxivTranslator
# from ..individual.brill import BrillTranslator
# from ..individual.cairn import CairnTranslator
# from ..individual.cambridge import CambridgeTranslator
# from ..individual.cambridge_core import CambridgeCoreTranslator
# from ..individual.cell_press import CellPressTranslator
# from ..individual.cern_document_server import CERNDocumentServerTranslator
# from ..individual.ceur_workshop_proceedings import (
#     CEURWorkshopProceedingsTranslator,
# )
# from ..individual.clacso import CLACSOTranslator
# from ..individual.csiro_publishing import CSIROPublishingTranslator
# from ..individual.dblp_computer_science_bibliography import DBLPTranslator
# from ..individual.digital_humanities_quarterly import (
#     DigitalHumanitiesQuarterlyTranslator,
# )
# from ..individual.dlibra import DLibraTranslator
# from ..individual.doi import DOITranslator
# from ..individual.e_periodica_switzerland import (
#     EPeriodicaSwitzerlandTranslator,
# )
# from ..individual.ebsco_discovery_layer import EBSCODiscoveryLayerTranslator
# from ..individual.elife import ELifeTranslator
# from ..individual.elsevier_health import ElsevierHealthTranslator
# from ..individual.elsevier_pure import ElsevierPureTranslator
# from ..individual.emerald import EmeraldTranslator
# from ..individual.europe_pmc import EuropePMCTranslator
# from ..individual.fachportal_padagogik import FachportalPadagogikTranslator
# from ..individual.frontiers import FrontiersTranslator
# from ..individual.gms_german_medical_science import (
#     GMSGermanMedicalScienceTranslator,
# )
# from ..individual.google_patents import GooglePatentsTranslator
# from ..individual.hindawi import HindawiTranslator
# from ..individual.ieee_computer_society import IEEEComputerSocietyTranslator
# from ..individual.ieee_xplore import IEEEXploreTranslator
# from ..individual.ietf import IETFTranslator
# from ..individual.ingenta_connect import IngentaConnectTranslator
# from ..individual.inter_research_science_center import (
#     InterResearchScienceCenterTranslator,
# )
# from ..individual.invenio_rdm import InvenioRDMTranslator
# from ..individual.iop import IOPTranslator
# from ..individual.jrc_publications_repository import (
#     JRCPublicationsRepositoryTranslator,
# )
# from ..individual.jstor import JSTORTranslator
# from ..individual.lingbuzz import LingBuzzTranslator
# from ..individual.lww import LWWTranslator
# from ..individual.mdpi import MDPITranslator
# from ..individual.medline_nbib import MEDLINEnbibTranslator
# from ..individual.nature import NatureTranslator
# from ..individual.nature_publishing_group import (
#     NaturePublishingGroupTranslator,
# )
# from ..individual.nber import NBERTranslator
# from ..individual.open_knowledge_repository import (
#     OpenKnowledgeRepositoryTranslator,
# )
# from ..individual.openalex_json import OpenAlexJSONTranslator
# from ..individual.openedition_journals import OpenEditionJournalsTranslator
# from ..individual.oxford import OxfordTranslator
# from ..individual.pkp_catalog_systems import PKPCatalogSystemsTranslator
# from ..individual.plos import PLoSTranslator
# from ..individual.project_muse import ProjectMUSETranslator
# from ..individual.pubfactory_journals import PubFactoryJournalsTranslator
# from ..individual.pubmed import PubMedTranslator
# from ..individual.pubmed_central import PubMedCentralTranslator
# from ..individual.pubmed_xml import PubMedXMLTranslator
# from ..individual.research_square import ResearchSquareTranslator
# from ..individual.rsc import RSCTranslator
# from ..individual.sage import SAGETranslator
# from ..individual.sage_journals import SAGEJournalsTranslator
# from ..individual.scholars_portal_journals import (
#     ScholarsPortalJournalsTranslator,
# )
# from ..individual.sciencedirect import ScienceDirectTranslator
# from ..individual.scinapse import ScinapseTranslator
# from ..individual.semantic_scholar import SemanticScholarTranslator
# from ..individual.silverchair import SilverchairTranslator
# from ..individual.springer import SpringerTranslator
# 
# # Import all translator implementations
# from ..individual.ssrn import SSRNTranslator
# from ..individual.state_records_office_wa import StateRecordsOfficeWATranslator
# from ..individual.superlib import SuperlibTranslator
# from ..individual.taylor_francis import TaylorFrancisTranslator
# from ..individual.taylor_francis_nejm import TaylorFrancisNEJMTranslator
# from ..individual.theory_of_computing import TheoryOfComputingTranslator
# from ..individual.tony_blair_institute import TonyBlairInstituteTranslator
# from ..individual.treesearch import TreesearchTranslator
# from ..individual.verniana import VernianaTranslator
# from ..individual.web_of_science import WebOfScienceTranslator
# from ..individual.who import WHOTranslator
# from ..individual.wiley import WileyTranslator
# from ..individual.wilson_center_digital_archive import (
#     WilsonCenterDigitalArchiveTranslator,
# )
# from ..individual.world_digital_library import WorldDigitalLibraryTranslator
# from ..individual.ypfs import YPFSTranslator
# from ..individual.zbmath import ZbMATHTranslator
# from ..individual.zobodat import ZOBODATTranslator
# 
# 
# class TranslatorRegistry:
#     """Central registry for all Python translator implementations.
# 
#     Translators are checked in order. The first matching translator wins.
#     DOI is first because it redirects to publisher pages and delegates to them.
#     """
# 
#     _translators: List[Type[BaseTranslator]] = [
#         # DOI first - it redirects to publishers and delegates
#         DOITranslator,
#         SSRNTranslator,
#         NatureTranslator,
#         NaturePublishingGroupTranslator,
#         ScienceDirectTranslator,
#         WileyTranslator,
#         IEEEXploreTranslator,
#         MDPITranslator,
#         ArXivTranslator,
#         BioRxivTranslator,
#         FrontiersTranslator,
#         PLoSTranslator,
#         SemanticScholarTranslator,
#         SilverchairTranslator,
#         SpringerTranslator,
#         PubMedTranslator,
#         PubMedCentralTranslator,
#         PubMedXMLTranslator,
#         MEDLINEnbibTranslator,
#         JSTORTranslator,
#         ACSTranslator,
#         BioMedCentralTranslator,
#         HindawiTranslator,
#         IOPTranslator,
#         OxfordTranslator,
#         TaylorFrancisTranslator,
#         TaylorFrancisNEJMTranslator,
#         CambridgeTranslator,
#         SAGETranslator,
#         EmeraldTranslator,
#         ResearchSquareTranslator,
#         CellPressTranslator,
#         EuropePMCTranslator,
#         AnnualReviewsTranslator,
#         SAGEJournalsTranslator,
#         # CambridgeCoreTranslator,  # Not yet implemented
#         ACSPublicationsTranslator,
#         ACMTranslator,
#         RSCTranslator,
#         BrillTranslator,
#         APSTranslator,
#         AIPTranslator,
#         # AtyponTranslator,  # Removed - use AtyponJournalsTranslator instead
#         BioOneTranslator,
#         ProjectMUSETranslator,
#         AMSJournalsTranslator,
#         WebOfScienceTranslator,
#         AEAWebTranslator,
#         ElsevierHealthTranslator,
#         ElsevierPureTranslator,
#         ASCETranslator,
#         LWWTranslator,
#         AccessEngineeringTranslator,
#         AccessScienceTranslator,
#         ACLSHumanitiesEBookTranslator,
#         ACLWebTranslator,
#         AdamMatthewDigitalTranslator,
#         IngentaConnectTranslator,
#         CairnTranslator,
#         DLibraTranslator,
#         FachportalPadagogikTranslator,
#         InvenioRDMTranslator,
#         NBERTranslator,
#         AOSICTranslator,
#         CambridgeCoreTranslator,
#         PubFactoryJournalsTranslator,
#         OpenKnowledgeRepositoryTranslator,
#         CERNDocumentServerTranslator,
#         DigitalHumanitiesQuarterlyTranslator,
#         WHOTranslator,
#         JRCPublicationsRepositoryTranslator,
#         EPeriodicaSwitzerlandTranslator,
#         GooglePatentsTranslator,
#         IETFTranslator,
#         # CSIROPublishingTranslator,  # File missing
#         CLACSOTranslator,
#         CEURWorkshopProceedingsTranslator,
#         WilsonCenterDigitalArchiveTranslator,
#         WorldDigitalLibraryTranslator,
#         ZOBODATTranslator,
#         VernianaTranslator,
#         StateRecordsOfficeWATranslator,
#         TreesearchTranslator,
#         TonyBlairInstituteTranslator,
#         TheoryOfComputingTranslator,
#         SuperlibTranslator,
#         YPFSTranslator,
#         APSPhysicsTranslator,
#         DBLPTranslator,
#         ZbMATHTranslator,
#         AtyponJournalsTranslator,
#         GMSGermanMedicalScienceTranslator,
#         IEEEComputerSocietyTranslator,
#         InterResearchScienceCenterTranslator,
#         ScinapseTranslator,
#         ELifeTranslator,
#         PKPCatalogSystemsTranslator,
#         OpenAlexJSONTranslator,
#         ScholarsPortalJournalsTranslator,
#         LingBuzzTranslator,
#         OpenEditionJournalsTranslator,
#         EBSCODiscoveryLayerTranslator,
#         AllAfricaTranslator,
#         ArXivOrgTranslator,
#         ABCNewsAustraliaTranslator,
#         ADSBibcodeTranslator,
#         AGRISTranslator,
#         # Add more translators here as they are implemented
#     ]
# 
#     def __init__(self):
#         self.name = self.__class__.__name__
# 
#     @classmethod
#     def get_translator_for_url(cls, url: str) -> Optional[Type[BaseTranslator]]:
#         """Find the appropriate translator for a given URL.
# 
#         Args:
#             url: URL to find translator for
# 
#         Returns:
#             Translator class if found, None otherwise
#         """
#         # Check if pattern-based extraction can handle it
#         try:
#             from .patterns import AccessPattern, detect_pattern
# 
#             pattern, _ = detect_pattern(url)
#             if pattern == AccessPattern.DIRECT_PDF:
#                 # Signal that we can handle it (patterns.py will extract)
#                 return type("DirectPDFTranslator", (), {"LABEL": "Direct PDF"})
#         except Exception:
#             pass
# 
#         for translator in cls._translators:
#             if translator.matches_url(url):
#                 return translator
#         return None
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, url: str, page: Page) -> List[str]:
#         """Extract PDF URLs using the appropriate translator.
# 
#         Args:
#             url: URL of the page
#             page: Playwright page object
# 
#         Returns:
#             List of PDF URLs found, or empty list if no translator found
#         """
#         # Try pattern-based extraction first
#         try:
#             from .patterns import extract_pdf_urls_by_pattern
# 
#             pdf_urls = await extract_pdf_urls_by_pattern(url, page)
#             if pdf_urls:
#                 return pdf_urls
#         except Exception as e:
#             logger.debug(f"{self.name}: Pattern extraction failed for {url}: {e}")
# 
#         # Fall back to translator-based approach
#         translator = cls.get_translator_for_url(url)
#         if translator:
#             try:
#                 return await translator.extract_pdf_urls_async(page)
#             except Exception as e:
#                 logger.error(
#                     f"{self.name}: Translator {translator.LABEL} failed for {url}: {e}"
#                 )
#                 return []
# 
#         logger.debug(f"{self.name}: No translator found for {url}")
#         return []
# 
#     @classmethod
#     def register(cls, translator: Type[BaseTranslator]) -> None:
#         """Register a new translator.
# 
#         Args:
#             translator: Translator class to register
#         """
#         if translator not in cls._translators:
#             cls._translators.append(translator)
# 
#     @classmethod
#     def list_translators(cls) -> List[Type[BaseTranslator]]:
#         """Get list of all registered translators.
# 
#         Returns:
#             List of translator classes
#         """
#         return cls._translators.copy()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/core/registry.py
# --------------------------------------------------------------------------------
