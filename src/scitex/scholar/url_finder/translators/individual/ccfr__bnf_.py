#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CCfr (BnF) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class CcfrBnfTranslator(BaseTranslator):
    """CCfr (BnF)."""

    LABEL = "CCfr (BnF)"
    URL_TARGET_PATTERN = r"^https?://ccfr\.bnf\.fr/portailccfr/.*\b(action=search|menu=menu_view_grappage|search\.jsp)\b"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
