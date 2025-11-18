#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HighWire translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class HighWireTranslator(BaseTranslator):
    """HighWire."""

    LABEL = "HighWire"
    URL_TARGET_PATTERN = r"^https?://[^/]+/(cgi/searchresults|cgi/search|cgi/content/(abstract|full|short|summary)|current\.dtl$|content/vol[0-9]+/issue[0-9]+/(index\.dtl)?$)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
