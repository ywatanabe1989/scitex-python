#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""National Archives of South Africa translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class NationalArchivesOfSouthAfricaTranslator(BaseTranslator):
    """National Archives of South Africa."""

    LABEL = "National Archives of South Africa"
    URL_TARGET_PATTERN = r"^https?://www\.national\.archsrch\.gov\.za"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
