#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARTFL Encyclopedie translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ARTFLEncyclopedieTranslator(BaseTranslator):
    """ARTFL Encyclopedie."""

    LABEL = "ARTFL Encyclopedie"
    URL_TARGET_PATTERN = r"^https?://artflsrv\d+\.uchicago\.edu/philologic4/encyclopedie\d+/(navigate/|query)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
