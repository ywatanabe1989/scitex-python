#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The Economic Times - The Times of India translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class TheEconomicTimesTheTimesOfIndiaTranslator(BaseTranslator):
    """The Economic Times - The Times of India."""

    LABEL = "The Economic Times - The Times of India"
    URL_TARGET_PATTERN = r"^https?://(economictimes|timesofindia)\.indiatimes\.com"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
