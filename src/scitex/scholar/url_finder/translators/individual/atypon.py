#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atypon Journals translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class AtyponTranslator(BaseTranslator):
    """Atypon Journals."""

    LABEL = "Atypon Journals"
    URL_TARGET_PATTERN = r"^https?://[^?#]+(/doi/((abs|abstract|full|figure|ref|citedby|book)/)?10\.|/action/doSearch\?)|^https?://[^/]+/toc/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
