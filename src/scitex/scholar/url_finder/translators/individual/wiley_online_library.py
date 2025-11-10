#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wiley Online Library translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class WileyOnlineLibraryTranslator(BaseTranslator):
    """Wiley Online Library."""

    LABEL = "Wiley Online Library"
    URL_TARGET_PATTERN = r"^https?://([\w-]+\.)?onlinelibrary\.wiley\.com[^/]*/(book|doi|toc|advanced/search|search-web/cochrane|cochranelibrary/search|o/cochrane/(clcentral|cldare|clcmr|clhta|cleed|clabout)/articles/.+/sect0\.html)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
