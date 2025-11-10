#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bibliotheque nationale de France translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class BibliothequeNationaleDeFranceTranslator(BaseTranslator):
    """Bibliotheque nationale de France."""

    LABEL = "Bibliotheque nationale de France"
    URL_TARGET_PATTERN = r"^https?://[^/]*catalogue\.bnf\.fr"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
