#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bibliotheque et Archives Nationales du Quebec translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class BibliothequeEtArchivesNationalesDuQuebecTranslator(BaseTranslator):
    """Bibliotheque et Archives Nationales du Quebec."""

    LABEL = "Bibliotheque et Archives Nationales du Quebec"
    URL_TARGET_PATTERN = r"^https?://catalogue\.banq\.qc\.ca/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
