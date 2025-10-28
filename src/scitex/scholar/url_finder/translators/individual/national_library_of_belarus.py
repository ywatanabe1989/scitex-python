#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""National Library of Belarus translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class NationalLibraryOfBelarusTranslator(BaseTranslator):
    """National Library of Belarus."""

    LABEL = "National Library of Belarus"
    URL_TARGET_PATTERN = r"^https?://www\.nlb\.by/portal/page/portal/index/resources/(basicsearch|expandedsearch|anothersearch|authoritet|newdoc|top100)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
