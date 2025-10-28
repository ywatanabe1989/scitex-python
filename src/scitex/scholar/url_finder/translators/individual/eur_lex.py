#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EUR-Lex translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class EurLexTranslator(BaseTranslator):
    """EUR-Lex."""

    LABEL = "EUR-Lex"
    URL_TARGET_PATTERN = r"^https?://(www\.)?eur-lex\.europa\.eu/(legal-content/[A-Z][A-Z]/(TXT|ALL)/|search\.html\?)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
