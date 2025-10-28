#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archiv fuer Sozialgeschichte translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ArchivFuerSozialgeschichteTranslator(BaseTranslator):
    """Archiv fuer Sozialgeschichte."""

    LABEL = "Archiv fuer Sozialgeschichte"
    URL_TARGET_PATTERN = r"^https?://library\.fes\.de/jportal/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
