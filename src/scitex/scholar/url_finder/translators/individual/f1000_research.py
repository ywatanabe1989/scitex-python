#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F1000 Research translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class F1000ResearchTranslator(BaseTranslator):
    """F1000 Research."""

    LABEL = "F1000 Research"
    URL_TARGET_PATTERN = r"^https?://(www\.)?(((openresearchcentral|(aas|amrc|hrb|wellcome|gates)openresearch)\.org)|(f1000research|emeraldopenresearch)\.com)/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
