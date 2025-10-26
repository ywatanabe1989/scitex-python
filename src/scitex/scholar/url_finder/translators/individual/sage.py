#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAGE Journals translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class SAGETranslator(BaseTranslator):
    """SAGE Journals."""

    LABEL = "SAGE Journals"
    URL_TARGET_PATTERN = r"^https?://journals\.sagepub\.com(/doi/((abs|full|pdf)/)?10\.|/action/doSearch\?|/toc/)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
