#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arXiv.org translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ArXivTranslator(BaseTranslator):
    """arXiv.org."""

    LABEL = "arXiv.org"
    URL_TARGET_PATTERN = r"^https?://([^.]+\.)?(arxiv\.org|xxx\.lanl\.gov)/(search|find|catchup|list/\w|abs/|pdf/)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from arXiv page.

        arXiv URLs follow a simple pattern:
        - Abstract: https://arxiv.org/abs/XXXX.XXXXX
        - PDF: https://arxiv.org/pdf/XXXX.XXXXX.pdf
        """
        url = page.url

        # Extract arXiv ID from URL
        # Matches: /abs/2308.09312 or /pdf/2308.09312
        match = re.search(r"/(abs|pdf)/(\d+\.\d+)", url)
        if match:
            arxiv_id = match.group(2)
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            return [pdf_url]

        return []
