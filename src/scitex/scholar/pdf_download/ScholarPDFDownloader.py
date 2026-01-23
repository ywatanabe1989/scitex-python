#!/usr/bin/env python3
# Timestamp: "2026-01-22 (ywatanabe)"
# File: src/scitex/scholar/pdf_download/ScholarPDFDownloader.py
"""PDF downloader with multiple fallback strategies."""

from __future__ import annotations

import asyncio
import os
import traceback
from pathlib import Path
from typing import List, Optional, Union

from playwright.async_api import BrowserContext

from scitex import logging
from scitex.scholar import ScholarConfig
from scitex.scholar.pdf_download.strategies import (
    FlexibleFilenameGenerator,
    handle_manual_download_on_page_async,
    try_download_chrome_pdf_viewer_async,
    try_download_direct_async,
    try_download_open_access_async,
    try_download_response_body_async,
)

logger = logging.getLogger(__name__)


class ScholarPDFDownloader:
    """Download PDFs from URLs with multiple fallback strategies.

    Strategies tried in order:
    - Chrome PDF Viewer
    - Direct Download (ERR_ABORTED)
    - Response Body Extraction
    - Manual Download Fallback

    URL resolution (DOI -> URL) should be handled by the caller.
    """

    def __init__(self, context: BrowserContext, config: ScholarConfig = None):
        self.name = self.__class__.__name__
        self.config = config if config else ScholarConfig()
        self.context = context
        self.output_dir = self.config.get_library_downloads_dir()

        self.prefer_open_access = self.config.resolve(
            "prefer_open_access", default=True, type=bool
        )
        self.enable_paywall_access = self.config.resolve(
            "enable_paywall_access", default=False, type=bool
        )
        self.track_paywall_attempts = self.config.resolve(
            "track_paywall_attempts", default=True, type=bool
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def download_from_urls(
        self,
        pdf_urls: List[str],
        output_dir: Union[str, Path] = None,
        max_concurrent: int = 3,
    ) -> List[Path]:
        """Download multiple PDFs with parallel processing."""
        output_dir = output_dir or self.output_dir
        if not pdf_urls:
            return []

        output_paths = [
            output_dir / f"{ii_pdf:03d}_{os.path.basename(pdf_url)}"
            for ii_pdf, pdf_url in enumerate(pdf_urls)
        ]

        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(url: str, path: Path, index: int):
            async with semaphore:
                logger.info(
                    f"{self.name}: Downloading PDF {index}/{len(pdf_urls)}: {url}"
                )
                result = await self.download_from_url(url, path)
                if result:
                    logger.info(f"{self.name}: Downloaded to {result}")
                return result

        tasks = [
            download_with_semaphore(url, path, idx + 1)
            for idx, (url, path) in enumerate(zip(pdf_urls, output_paths))
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        saved_paths = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"{self.name}: Download error: {result}")
            elif result:
                saved_paths.append(result)

        logger.info(f"{self.name}: Downloaded {len(saved_paths)}/{len(pdf_urls)} PDFs")
        return saved_paths

    async def download_open_access(
        self,
        oa_url: str,
        output_path: Union[str, Path],
        metadata: Optional[dict] = None,
    ) -> Optional[Path]:
        """Download PDF from an Open Access URL."""
        if not oa_url:
            logger.debug(f"{self.name}: No OA URL provided")
            return None

        if isinstance(output_path, str):
            output_path = Path(output_path)
        if not str(output_path).endswith(".pdf"):
            output_path = Path(str(output_path) + ".pdf")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"{self.name}: Attempting OA download from {oa_url[:60]}...")

        result = await try_download_open_access_async(
            oa_url=oa_url,
            output_path=output_path,
            metadata=metadata,
            func_name=self.name,
        )

        if result:
            logger.info(f"{self.name}: Successfully downloaded OA PDF to {result}")
        else:
            logger.debug(f"{self.name}: OA download failed")

        return result

    async def download_smart(
        self, paper, output_path: Union[str, Path]
    ) -> Optional[Path]:
        """Smart download choosing best strategy based on paper metadata."""
        if isinstance(output_path, str):
            output_path = Path(output_path)
        if not str(output_path).endswith(".pdf"):
            output_path = Path(str(output_path) + ".pdf")

        meta = paper.metadata if hasattr(paper, "metadata") else paper
        access = getattr(meta, "access", None)
        url_meta = getattr(meta, "url", None)
        id_meta = getattr(meta, "id", None)

        is_open_access = getattr(access, "is_open_access", False) if access else False
        oa_url = getattr(access, "oa_url", None) if access else None
        pdf_urls = getattr(url_meta, "pdfs", []) if url_meta else []
        doi = getattr(id_meta, "doi", None) if id_meta else None

        logger.info(f"{self.name}: Smart download for DOI={doi}, OA={is_open_access}")

        # Strategy 1: Try Open Access if available
        if self.prefer_open_access and oa_url:
            logger.info(f"{self.name}: Trying Open Access URL first")
            result = await self.download_open_access(oa_url, output_path)
            if result:
                if access and self.track_paywall_attempts:
                    access.paywall_bypass_attempted = False
                return result

        # Strategy 2: Try available PDF URLs
        for pdf_entry in pdf_urls:
            pdf_url = pdf_entry.get("url") if isinstance(pdf_entry, dict) else pdf_entry
            if pdf_url:
                logger.info(f"{self.name}: Trying PDF URL: {pdf_url[:60]}...")
                result = await self.download_from_url(pdf_url, output_path, doi=doi)
                if result:
                    return result

        # Strategy 3: Try paywall access if enabled
        if self.enable_paywall_access and not is_open_access:
            logger.info(f"{self.name}: Attempting paywall access (opt-in enabled)")
            if access and self.track_paywall_attempts:
                access.paywall_bypass_attempted = True

            if doi:
                doi_url = f"https://doi.org/{doi}"
                result = await self.download_from_url(doi_url, output_path, doi=doi)
                if result:
                    if access and self.track_paywall_attempts:
                        access.paywall_bypass_success = True
                    return result
                elif access and self.track_paywall_attempts:
                    access.paywall_bypass_success = False

        logger.warning(f"{self.name}: All download strategies exhausted for DOI={doi}")
        return None

    async def download_from_url(
        self,
        pdf_url: str,
        output_path: Union[str, Path],
        doi: Optional[str] = None,
    ) -> Optional[Path]:
        """Main download method with manual override support."""
        if not pdf_url:
            logger.warning(f"{self.name}: PDF URL passed but not valid: {pdf_url}")
            return None

        if isinstance(output_path, str):
            output_path = Path(output_path)
        if not str(output_path).endswith(".pdf"):
            output_path = Path(str(output_path) + ".pdf")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_filename = FlexibleFilenameGenerator.generate_filename(
            doi=doi, url=pdf_url, content_type="main"
        )

        stop_event = asyncio.Event()
        self.context._scitex_is_manual_mode = False
        self.context._scitex_manual_mode_event = stop_event

        from scitex.scholar.pdf_download.strategies.manual_download_utils import (
            get_manual_button_init_script,
        )

        button_script = get_manual_button_init_script(target_filename)
        await self.context.add_init_script(button_script)
        logger.info(f"{self.name}: Manual mode button injected into browser context")

        button_task = None
        pdf_page = None

        async def chrome_pdf_wrapper(url, path):
            return await try_download_chrome_pdf_viewer_async(
                self.context, url, path, self.name
            )

        async def direct_download_wrapper(url, path):
            return await try_download_direct_async(self.context, url, path, self.name)

        async def response_body_wrapper(url, path):
            return await try_download_response_body_async(
                self.context, url, path, self.name
            )

        async def manual_fallback_wrapper(url, path):
            return None

        try_download_methods = [
            ("Chrome PDF", chrome_pdf_wrapper),
            ("Direct Download", direct_download_wrapper),
            ("From Response Body", response_body_wrapper),
            ("Manual Download", manual_fallback_wrapper),
        ]

        for method_name, method_func in try_download_methods:
            if stop_event.is_set():
                logger.info(f"{self.name}: Manual mode - stopping automation")
                break

            logger.info(f"{self.name}: Trying method: {method_name}")

            try:
                if stop_event.is_set():
                    logger.info(f"{self.name}: Manual mode, skipping {method_name}")
                    break

                is_downloaded = await method_func(pdf_url, output_path)

                if stop_event.is_set():
                    logger.info(f"{self.name}: Manual mode during {method_name}")
                    break

                if is_downloaded:
                    if button_task:
                        button_task.cancel()
                    if pdf_page:
                        await pdf_page.close()
                    logger.info(f"{self.name}: Downloaded via {method_name}")
                    return is_downloaded
                else:
                    logger.debug(f"{self.name}: {method_name} returned None")
            except Exception as e:
                logger.warning(f"{self.name}: {method_name} raised exception: {e}")
                logger.debug(f"{self.name}: Traceback: {traceback.format_exc()}")

        # Handle manual download if user chose it
        if stop_event.is_set():
            self.context._scitex_is_manual_mode = True
            logger.info(f"{self.name}: User chose manual download - starting")
            if button_task:
                button_task.cancel()

            if not pdf_page:
                pdf_page = await self.context.new_page()
                await pdf_page.goto(
                    pdf_url, timeout=30000, wait_until="domcontentloaded"
                )

            result = await handle_manual_download_on_page_async(
                pdf_page,
                pdf_url,
                output_path,
                func_name=self.name,
                config=self.config,
                doi=doi,
            )
            await pdf_page.close()
            return result

        # All methods failed
        if button_task:
            button_task.cancel()
        if pdf_page:
            await pdf_page.close()
        logger.fail(f"{self.name}: All download methods failed for {pdf_url}")
        return None


# CLI entry point moved to _cli.py
if __name__ == "__main__":
    from scitex.scholar.pdf_download._cli import run_main

    run_main()

# EOF
