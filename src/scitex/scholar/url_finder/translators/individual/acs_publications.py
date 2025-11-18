#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACS Publications translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ACSPublicationsTranslator(BaseTranslator):
    """ACS Publications."""

    LABEL = "ACS Publications"
    URL_TARGET_PATTERN = r"^https?://pubs\.acs\.org/(toc/|journal/|topic/|isbn/\d|doi/(full/|abs/|epdf/|book/)?10\.|action/(doSearch\?|showCitFormats\?.*doi))"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
