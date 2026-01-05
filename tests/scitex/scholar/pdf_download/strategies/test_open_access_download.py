# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/pdf_download/strategies/open_access_download.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: ./src/scitex/scholar/pdf_download/strategies/open_access_download.py
# """
# Open Access PDF Download Strategy.
# 
# Downloads PDFs from known Open Access sources with appropriate handling
# for each source type (arXiv, PubMed Central, OpenAlex OA URLs, etc.).
# """
# 
# from pathlib import Path
# from typing import Optional, Dict, Any
# import aiohttp
# 
# from scitex import logging
# 
# logger = logging.getLogger(__name__)
# 
# 
# # Known OA source patterns and their handlers
# OA_SOURCE_PATTERNS = {
#     "arxiv": {
#         "patterns": ["arxiv.org"],
#         "pdf_transform": lambda url: url.replace("/abs/", "/pdf/") + ".pdf"
#         if "/abs/" in url
#         else url,
#     },
#     "pmc": {
#         "patterns": ["ncbi.nlm.nih.gov/pmc", "europepmc.org"],
#         "pdf_transform": lambda url: url,  # PMC links are usually direct
#     },
#     "biorxiv": {
#         "patterns": ["biorxiv.org", "medrxiv.org"],
#         "pdf_transform": lambda url: url + ".full.pdf"
#         if not url.endswith(".pdf")
#         else url,
#     },
#     "doaj": {
#         "patterns": ["doaj.org"],
#         "pdf_transform": lambda url: url,
#     },
#     "zenodo": {
#         "patterns": ["zenodo.org"],
#         "pdf_transform": lambda url: url,
#     },
# }
# 
# 
# def _identify_oa_source(url: str) -> Optional[str]:
#     """Identify which OA source a URL belongs to."""
#     url_lower = url.lower()
#     for source_name, config in OA_SOURCE_PATTERNS.items():
#         for pattern in config["patterns"]:
#             if pattern in url_lower:
#                 return source_name
#     return None
# 
# 
# def _transform_to_pdf_url(url: str, source: str) -> str:
#     """Transform URL to direct PDF URL based on source."""
#     if source in OA_SOURCE_PATTERNS:
#         transform_func = OA_SOURCE_PATTERNS[source]["pdf_transform"]
#         return transform_func(url)
#     return url
# 
# 
# async def try_download_open_access_async(
#     oa_url: str,
#     output_path: Path,
#     metadata: Optional[Dict[str, Any]] = None,
#     func_name: str = "try_download_open_access_async",
#     timeout: int = 60,
# ) -> Optional[Path]:
#     """
#     Download PDF from an Open Access URL.
# 
#     This strategy is simpler than browser-based strategies because OA PDFs
#     are typically directly accessible without authentication.
# 
#     Args:
#         oa_url: Open Access URL (from OpenAlex oa_url, arXiv, PMC, etc.)
#         output_path: Path to save the downloaded PDF
#         metadata: Optional paper metadata for logging
#         func_name: Function name for logging
#         timeout: Download timeout in seconds
# 
#     Returns:
#         Path to downloaded PDF if successful, None otherwise
#     """
#     if not oa_url:
#         logger.debug(f"{func_name}: No OA URL provided")
#         return None
# 
#     # Identify source and transform URL if needed
#     source = _identify_oa_source(oa_url)
#     pdf_url = _transform_to_pdf_url(oa_url, source) if source else oa_url
# 
#     logger.info(
#         f"{func_name}: Attempting OA download from {source or 'unknown'}: {pdf_url[:80]}..."
#     )
# 
#     try:
#         # Create output directory if needed
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
# 
#         # Use aiohttp for async download
#         async with aiohttp.ClientSession() as session:
#             headers = {
#                 "User-Agent": "SciTeX/1.0 (Academic Research Tool; mailto:contact@scitex.io)",
#                 "Accept": "application/pdf,*/*",
#             }
# 
#             async with session.get(
#                 pdf_url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
#             ) as response:
#                 if response.status != 200:
#                     logger.warning(
#                         f"{func_name}: HTTP {response.status} from {pdf_url}"
#                     )
#                     return None
# 
#                 content_type = response.headers.get("Content-Type", "")
# 
#                 # Verify we're getting a PDF
#                 if "pdf" not in content_type.lower() and not pdf_url.endswith(".pdf"):
#                     # Some servers don't set content-type correctly, check magic bytes
#                     first_bytes = await response.content.read(5)
#                     if first_bytes != b"%PDF-":
#                         logger.warning(
#                             f"{func_name}: Response is not a PDF (content-type: {content_type})"
#                         )
#                         return None
#                     # Reset for full download
#                     content = first_bytes + await response.content.read()
#                 else:
#                     content = await response.read()
# 
#                 # Validate PDF content
#                 if len(content) < 1000:  # PDF should be at least 1KB
#                     logger.warning(
#                         f"{func_name}: Downloaded content too small ({len(content)} bytes)"
#                     )
#                     return None
# 
#                 if not content.startswith(b"%PDF-"):
#                     logger.warning(
#                         f"{func_name}: Downloaded content is not a valid PDF"
#                     )
#                     return None
# 
#                 # Save to file
#                 with open(output_path, "wb") as f:
#                     f.write(content)
# 
#                 size_mb = len(content) / 1024 / 1024
#                 logger.info(
#                     f"{func_name}: Successfully downloaded {size_mb:.2f} MB to {output_path}"
#                 )
#                 return output_path
# 
#     except aiohttp.ClientError as e:
#         logger.warning(f"{func_name}: HTTP client error: {e}")
#         return None
#     except TimeoutError:
#         logger.warning(f"{func_name}: Download timed out after {timeout}s")
#         return None
#     except Exception as e:
#         logger.error(f"{func_name}: Download failed: {e}")
#         return None
# 
# 
# def try_download_open_access_sync(
#     oa_url: str,
#     output_path: Path,
#     metadata: Optional[Dict[str, Any]] = None,
#     timeout: int = 60,
# ) -> Optional[Path]:
#     """
#     Synchronous wrapper for try_download_open_access_async.
# 
#     Args:
#         oa_url: Open Access URL
#         output_path: Path to save the downloaded PDF
#         metadata: Optional paper metadata
#         timeout: Download timeout in seconds
# 
#     Returns:
#         Path to downloaded PDF if successful, None otherwise
#     """
#     import asyncio
# 
#     try:
#         loop = asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
# 
#     return loop.run_until_complete(
#         try_download_open_access_async(oa_url, output_path, metadata, timeout=timeout)
#     )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/pdf_download/strategies/open_access_download.py
# --------------------------------------------------------------------------------
