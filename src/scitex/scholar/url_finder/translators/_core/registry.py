#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-10 01:11:16 (ywatanabe)"
# File: /home/ywatanabe/proj/zotero-translators-python/src/zotero_translators_python/core/registry.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/zotero_translators_python/core/registry.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""Translator registry for managing and discovering translators.

The registry finds the right translator for a URL and extracts PDF links.

Usage:
    from scitex.scholar.url_finder.translators import TranslatorRegistry

    # Find translator
    translator = TranslatorRegistry.get_translator_for_url(url)

    # Extract PDFs
    pdf_urls = await TranslatorRegistry.extract_pdf_urls_async(url, page)
"""

import logging
from typing import List, Optional, Type

from playwright.async_api import Page

from .base import BaseTranslator

logger = logging.getLogger(__name__)

from .._individual.abc_news_australia import ABCNewsAustraliaTranslator
from .._individual.access_engineering import AccessEngineeringTranslator
from .._individual.access_science import AccessScienceTranslator
from .._individual.acls_humanities_ebook import ACLSHumanitiesEBookTranslator
from .._individual.aclweb import ACLWebTranslator
from .._individual.acm import ACMTranslator
from .._individual.acs import ACSTranslator

# from .._individual.cambridge_core import CambridgeCoreTranslator  # Not yet implemented
from .._individual.acs_publications import ACSPublicationsTranslator
from .._individual.adam_matthew_digital import AdamMatthewDigitalTranslator
from .._individual.ads_bibcode import ADSBibcodeTranslator
from .._individual.aea_web import AEAWebTranslator
from .._individual.agris import AGRISTranslator
from .._individual.aip import AIPTranslator
from .._individual.all_africa import AllAfricaTranslator
from .._individual.ams_journals import AMSJournalsTranslator
from .._individual.annual_reviews import AnnualReviewsTranslator
from .._individual.aosic import AOSICTranslator
from .._individual.aps import APSTranslator
from .._individual.aps_physics import APSPhysicsTranslator
from .._individual.arxiv import ArXivTranslator
from .._individual.arxiv_org import ArXivOrgTranslator
from .._individual.asce import ASCETranslator

# from .._individual.atypon import AtyponTranslator  # Deprecated - use AtyponJournalsTranslator
from .._individual.atypon_journals import AtyponJournalsTranslator
from .._individual.biomed_central import BioMedCentralTranslator
from .._individual.bioone import BioOneTranslator
from .._individual.biorxiv import BioRxivTranslator
from .._individual.brill import BrillTranslator
from .._individual.cairn import CairnTranslator
from .._individual.cambridge import CambridgeTranslator
from .._individual.cambridge_core import CambridgeCoreTranslator
from .._individual.cell_press import CellPressTranslator
from .._individual.cern_document_server import CERNDocumentServerTranslator
from .._individual.ceur_workshop_proceedings import (
    CEURWorkshopProceedingsTranslator,
)
from .._individual.clacso import CLACSOTranslator
from .._individual.csiro_publishing import CSIROPublishingTranslator
from .._individual.dblp_computer_science_bibliography import DBLPTranslator
from .._individual.digital_humanities_quarterly import (
    DigitalHumanitiesQuarterlyTranslator,
)
from .._individual.dlibra import DLibraTranslator
from .._individual.doi import DOITranslator
from .._individual.e_periodica_switzerland import (
    EPeriodicaSwitzerlandTranslator,
)
from .._individual.ebsco_discovery_layer import EBSCODiscoveryLayerTranslator
from .._individual.elife import ELifeTranslator
from .._individual.elsevier_health import ElsevierHealthTranslator
from .._individual.elsevier_pure import ElsevierPureTranslator
from .._individual.emerald import EmeraldTranslator
from .._individual.europe_pmc import EuropePMCTranslator
from .._individual.fachportal_padagogik import FachportalPadagogikTranslator
from .._individual.frontiers import FrontiersTranslator
from .._individual.gms_german_medical_science import (
    GMSGermanMedicalScienceTranslator,
)
from .._individual.google_patents import GooglePatentsTranslator
from .._individual.hindawi import HindawiTranslator
from .._individual.ieee_computer_society import IEEEComputerSocietyTranslator
from .._individual.ieee_xplore import IEEEXploreTranslator
from .._individual.ietf import IETFTranslator
from .._individual.ingenta_connect import IngentaConnectTranslator
from .._individual.inter_research_science_center import (
    InterResearchScienceCenterTranslator,
)
from .._individual.invenio_rdm import InvenioRDMTranslator
from .._individual.iop import IOPTranslator
from .._individual.jrc_publications_repository import (
    JRCPublicationsRepositoryTranslator,
)
from .._individual.jstor import JSTORTranslator
from .._individual.lingbuzz import LingBuzzTranslator
from .._individual.lww import LWWTranslator
from .._individual.mdpi import MDPITranslator
from .._individual.medline_nbib import MEDLINEnbibTranslator
from .._individual.nature import NatureTranslator
from .._individual.nature_publishing_group import (
    NaturePublishingGroupTranslator,
)
from .._individual.nber import NBERTranslator
from .._individual.open_knowledge_repository import (
    OpenKnowledgeRepositoryTranslator,
)
from .._individual.openalex_json import OpenAlexJSONTranslator
from .._individual.openedition_journals import OpenEditionJournalsTranslator
from .._individual.oxford import OxfordTranslator
from .._individual.pkp_catalog_systems import PKPCatalogSystemsTranslator
from .._individual.plos import PLoSTranslator
from .._individual.project_muse import ProjectMUSETranslator
from .._individual.pubfactory_journals import PubFactoryJournalsTranslator
from .._individual.pubmed import PubMedTranslator
from .._individual.pubmed_central import PubMedCentralTranslator
from .._individual.pubmed_xml import PubMedXMLTranslator
from .._individual.research_square import ResearchSquareTranslator
from .._individual.rsc import RSCTranslator
from .._individual.sage import SAGETranslator
from .._individual.sage_journals import SAGEJournalsTranslator
from .._individual.scholars_portal_journals import (
    ScholarsPortalJournalsTranslator,
)
from .._individual.sciencedirect import ScienceDirectTranslator
from .._individual.scinapse import ScinapseTranslator
from .._individual.semantic_scholar import SemanticScholarTranslator
from .._individual.silverchair import SilverchairTranslator
from .._individual.springer import SpringerTranslator

# Import all translator implementations
from .._individual.ssrn import SSRNTranslator
from .._individual.state_records_office_wa import StateRecordsOfficeWATranslator
from .._individual.superlib import SuperlibTranslator
from .._individual.taylor_francis import TaylorFrancisTranslator
from .._individual.taylor_francis_nejm import TaylorFrancisNEJMTranslator
from .._individual.theory_of_computing import TheoryOfComputingTranslator
from .._individual.tony_blair_institute import TonyBlairInstituteTranslator
from .._individual.treesearch import TreesearchTranslator
from .._individual.verniana import VernianaTranslator
from .._individual.web_of_science import WebOfScienceTranslator
from .._individual.who import WHOTranslator
from .._individual.wiley import WileyTranslator
from .._individual.wilson_center_digital_archive import (
    WilsonCenterDigitalArchiveTranslator,
)
from .._individual.world_digital_library import WorldDigitalLibraryTranslator
from .._individual.ypfs import YPFSTranslator
from .._individual.zbmath import ZbMATHTranslator
from .._individual.zobodat import ZOBODATTranslator


class TranslatorRegistry:
    """Central registry for all Python translator implementations.

    Translators are checked in order. The first matching translator wins.
    DOI is first because it redirects to publisher pages and delegates to them.
    """

    _translators: List[Type[BaseTranslator]] = [
        # DOI first - it redirects to publishers and delegates
        DOITranslator,
        SSRNTranslator,
        NatureTranslator,
        NaturePublishingGroupTranslator,
        ScienceDirectTranslator,
        WileyTranslator,
        IEEEXploreTranslator,
        MDPITranslator,
        ArXivTranslator,
        BioRxivTranslator,
        FrontiersTranslator,
        PLoSTranslator,
        SemanticScholarTranslator,
        SilverchairTranslator,
        SpringerTranslator,
        PubMedTranslator,
        PubMedCentralTranslator,
        PubMedXMLTranslator,
        MEDLINEnbibTranslator,
        JSTORTranslator,
        ACSTranslator,
        BioMedCentralTranslator,
        HindawiTranslator,
        IOPTranslator,
        OxfordTranslator,
        TaylorFrancisTranslator,
        TaylorFrancisNEJMTranslator,
        CambridgeTranslator,
        SAGETranslator,
        EmeraldTranslator,
        ResearchSquareTranslator,
        CellPressTranslator,
        EuropePMCTranslator,
        AnnualReviewsTranslator,
        SAGEJournalsTranslator,
        # CambridgeCoreTranslator,  # Not yet implemented
        ACSPublicationsTranslator,
        ACMTranslator,
        RSCTranslator,
        BrillTranslator,
        APSTranslator,
        AIPTranslator,
        # AtyponTranslator,  # Removed - use AtyponJournalsTranslator instead
        BioOneTranslator,
        ProjectMUSETranslator,
        AMSJournalsTranslator,
        WebOfScienceTranslator,
        AEAWebTranslator,
        ElsevierHealthTranslator,
        ElsevierPureTranslator,
        ASCETranslator,
        LWWTranslator,
        AccessEngineeringTranslator,
        AccessScienceTranslator,
        ACLSHumanitiesEBookTranslator,
        ACLWebTranslator,
        AdamMatthewDigitalTranslator,
        IngentaConnectTranslator,
        CairnTranslator,
        DLibraTranslator,
        FachportalPadagogikTranslator,
        InvenioRDMTranslator,
        NBERTranslator,
        AOSICTranslator,
        CambridgeCoreTranslator,
        PubFactoryJournalsTranslator,
        OpenKnowledgeRepositoryTranslator,
        CERNDocumentServerTranslator,
        DigitalHumanitiesQuarterlyTranslator,
        WHOTranslator,
        JRCPublicationsRepositoryTranslator,
        EPeriodicaSwitzerlandTranslator,
        GooglePatentsTranslator,
        IETFTranslator,
        # CSIROPublishingTranslator,  # File missing
        CLACSOTranslator,
        CEURWorkshopProceedingsTranslator,
        WilsonCenterDigitalArchiveTranslator,
        WorldDigitalLibraryTranslator,
        ZOBODATTranslator,
        VernianaTranslator,
        StateRecordsOfficeWATranslator,
        TreesearchTranslator,
        TonyBlairInstituteTranslator,
        TheoryOfComputingTranslator,
        SuperlibTranslator,
        YPFSTranslator,
        APSPhysicsTranslator,
        DBLPTranslator,
        ZbMATHTranslator,
        AtyponJournalsTranslator,
        GMSGermanMedicalScienceTranslator,
        IEEEComputerSocietyTranslator,
        InterResearchScienceCenterTranslator,
        ScinapseTranslator,
        ELifeTranslator,
        PKPCatalogSystemsTranslator,
        OpenAlexJSONTranslator,
        ScholarsPortalJournalsTranslator,
        LingBuzzTranslator,
        OpenEditionJournalsTranslator,
        EBSCODiscoveryLayerTranslator,
        AllAfricaTranslator,
        ArXivOrgTranslator,
        ABCNewsAustraliaTranslator,
        ADSBibcodeTranslator,
        AGRISTranslator,
        # Add more translators here as they are implemented
    ]

    def __init__(self):
        self.name = self.__class__.__name__

    @classmethod
    def get_translator_for_url(cls, url: str) -> Optional[Type[BaseTranslator]]:
        """Find the appropriate translator for a given URL.

        Args:
            url: URL to find translator for

        Returns:
            Translator class if found, None otherwise
        """
        # Check if pattern-based extraction can handle it
        try:
            from .patterns import AccessPattern, detect_pattern

            pattern, _ = detect_pattern(url)
            if pattern == AccessPattern.DIRECT_PDF:
                # Signal that we can handle it (patterns.py will extract)
                return type("DirectPDFTranslator", (), {"LABEL": "Direct PDF"})
        except Exception:
            pass

        for translator in cls._translators:
            if translator.matches_url(url):
                return translator
        return None

    @classmethod
    async def extract_pdf_urls_async(cls, url: str, page: Page) -> List[str]:
        """Extract PDF URLs using the appropriate translator.

        Args:
            url: URL of the page
            page: Playwright page object

        Returns:
            List of PDF URLs found, or empty list if no translator found
        """
        # Try pattern-based extraction first
        try:
            from .patterns import extract_pdf_urls_by_pattern

            pdf_urls = await extract_pdf_urls_by_pattern(url, page)
            if pdf_urls:
                return pdf_urls
        except Exception as e:
            logger.debug(f"{self.name}: Pattern extraction failed for {url}: {e}")

        # Fall back to translator-based approach
        translator = cls.get_translator_for_url(url)
        if translator:
            try:
                return await translator.extract_pdf_urls_async(page)
            except Exception as e:
                logger.error(
                    f"{self.name}: Translator {translator.LABEL} failed for {url}: {e}"
                )
                return []

        logger.debug(f"{self.name}: No translator found for {url}")
        return []

    @classmethod
    def register(cls, translator: Type[BaseTranslator]) -> None:
        """Register a new translator.

        Args:
            translator: Translator class to register
        """
        if translator not in cls._translators:
            cls._translators.append(translator)

    @classmethod
    def list_translators(cls) -> List[Type[BaseTranslator]]:
        """Get list of all registered translators.

        Returns:
            List of translator classes
        """
        return cls._translators.copy()


# EOF
