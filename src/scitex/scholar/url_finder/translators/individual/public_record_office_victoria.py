#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Record Office Victoria translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class PublicRecordOfficeVictoriaTranslator(BaseTranslator):
    """Public Record Office Victoria."""

    LABEL = "Public Record Office Victoria"
    URL_TARGET_PATTERN = r"^https?://prov\.vic\.gov\.au/(archive|search_journey)/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
