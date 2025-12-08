#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nagoya University OPAC translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class NagoyaUniversityOpacTranslator(BaseTranslator):
    """Nagoya University OPAC."""

    LABEL = "Nagoya University OPAC"
    URL_TARGET_PATTERN = (
        r"^https?://opac\.nul\.nagoya-u\.ac\.jp/webopac/(catdbl\.do|ctlsrh\.do)"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
