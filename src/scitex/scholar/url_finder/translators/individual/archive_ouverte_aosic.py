#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archive Ouverte en Sciences de l'Information et de la Communication (AOSIC) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ArchiveOuverteAOSICTranslator(BaseTranslator):
    """Archive Ouverte en Sciences de l'Information et de la Communication (AOSIC)."""

    LABEL = (
        "Archive Ouverte en Sciences de l'Information et de la Communication (AOSIC)"
    )
    URL_TARGET_PATTERN = r"^https?://archivesic\.ccsd\.cnrs\.fr/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
