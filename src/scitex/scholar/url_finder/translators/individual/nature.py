#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nature Publishing Group translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class NatureTranslator(BaseTranslator):
    """Nature Publishing Group."""

    LABEL = "Nature Publishing Group"
    URL_TARGET_PATTERN = r"^https?://(www\.)?nature\.com/([^?/]+/)?(journal|archive|research|topten|search|full|abs|current_issue\.htm|most\.htm|articles/)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
