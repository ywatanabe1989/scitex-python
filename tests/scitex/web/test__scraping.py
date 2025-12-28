#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./tests/scitex/web/test__scraping.py

"""
Tests for web scraping utilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import shutil
from pathlib import Path
import re


class TestGetUrls:
    """Test get_urls function."""

    @patch('scitex.web._scraping.requests.get')
    def test_get_urls_basic(self, mock_get):
        """Test basic URL extraction."""
        from scitex.web import get_urls

        mock_response = Mock()
        mock_response.text = '''
        <html>
            <body>
                <a href="https://example.com/page1">Link 1</a>
                <a href="/page2">Link 2</a>
                <a href="https://example.com/page3">Link 3</a>
            </body>
        </html>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = get_urls('https://example.com')

        assert len(urls) == 3
        assert 'https://example.com/page1' in urls
        assert 'https://example.com/page2' in urls
        assert 'https://example.com/page3' in urls

    @patch('scitex.web._scraping.requests.get')
    def test_get_urls_with_pattern(self, mock_get):
        """Test URL extraction with pattern filtering."""
        from scitex.web import get_urls

        mock_response = Mock()
        mock_response.text = '''
        <html>
            <body>
                <a href="https://example.com/doc.pdf">PDF</a>
                <a href="https://example.com/page.html">HTML</a>
                <a href="https://example.com/data.pdf">Another PDF</a>
            </body>
        </html>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = get_urls('https://example.com', pattern=r'\.pdf$')

        assert len(urls) == 2
        assert all(url.endswith('.pdf') for url in urls)

    @patch('scitex.web._scraping.requests.get')
    def test_get_urls_same_domain(self, mock_get):
        """Test URL extraction with same domain filter."""
        from scitex.web import get_urls

        mock_response = Mock()
        mock_response.text = '''
        <html>
            <body>
                <a href="https://example.com/page1">Internal</a>
                <a href="https://other.com/page2">External</a>
                <a href="/page3">Relative</a>
            </body>
        </html>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = get_urls('https://example.com', same_domain=True)

        assert len(urls) == 2
        assert all('example.com' in url for url in urls)
        assert not any('other.com' in url for url in urls)

    @patch('scitex.web._scraping.requests.get')
    def test_get_urls_relative_urls(self, mock_get):
        """Test conversion of relative URLs to absolute."""
        from scitex.web import get_urls

        mock_response = Mock()
        mock_response.text = '''
        <html>
            <body>
                <a href="/page1">Page 1</a>
                <a href="page2">Page 2</a>
                <a href="../page3">Page 3</a>
            </body>
        </html>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = get_urls('https://example.com/dir/', absolute=True)

        assert len(urls) == 3
        assert all(url.startswith('https://') for url in urls)

    @patch('scitex.web._scraping.requests.get')
    def test_get_urls_request_failure(self, mock_get):
        """Test handling of request failures."""
        from scitex.web import get_urls
        import requests

        mock_get.side_effect = requests.RequestException("Network error")

        urls = get_urls('https://example.com')

        assert urls == []

    @patch('scitex.web._scraping.requests.get')
    def test_get_urls_duplicate_removal(self, mock_get):
        """Test that duplicate URLs are removed."""
        from scitex.web import get_urls

        mock_response = Mock()
        mock_response.text = '''
        <html>
            <body>
                <a href="https://example.com/page1">Link 1</a>
                <a href="https://example.com/page1">Link 1 again</a>
                <a href="/page1">Relative to same page</a>
            </body>
        </html>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = get_urls('https://example.com')

        # Should only have one instance of page1
        assert len(urls) == 1

    @patch('scitex.web._scraping.requests.get')
    def test_get_urls_empty_page(self, mock_get):
        """Test handling of page with no links."""
        from scitex.web import get_urls

        mock_response = Mock()
        mock_response.text = '<html><body>No links here</body></html>'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = get_urls('https://example.com')

        assert urls == []


class TestGetImageUrls:
    """Test get_image_urls function."""

    @patch('scitex.web._scraping.requests.get')
    def test_get_image_urls_basic(self, mock_get):
        """Test basic image URL extraction."""
        from scitex.web import get_image_urls

        mock_response = Mock()
        mock_response.text = '''
        <html>
            <body>
                <img src="https://example.com/image1.jpg">
                <img src="/images/image2.png">
                <img src="https://example.com/image3.gif">
            </body>
        </html>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        img_urls = get_image_urls('https://example.com')

        assert len(img_urls) == 3
        assert 'https://example.com/image1.jpg' in img_urls
        assert 'https://example.com/images/image2.png' in img_urls

    @patch('scitex.web._scraping.requests.get')
    def test_get_image_urls_with_pattern(self, mock_get):
        """Test image URL extraction with pattern filtering."""
        from scitex.web import get_image_urls

        mock_response = Mock()
        mock_response.text = '''
        <html>
            <body>
                <img src="https://example.com/image1.jpg">
                <img src="https://example.com/image2.png">
                <img src="https://example.com/image3.jpg">
            </body>
        </html>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        img_urls = get_image_urls('https://example.com', pattern=r'\.jpg$')

        assert len(img_urls) == 2
        assert all(url.endswith('.jpg') for url in img_urls)

    @patch('scitex.web._scraping.requests.get')
    def test_get_image_urls_same_domain(self, mock_get):
        """Test image URL extraction with same domain filter."""
        from scitex.web import get_image_urls

        mock_response = Mock()
        mock_response.text = '''
        <html>
            <body>
                <img src="https://example.com/image1.jpg">
                <img src="https://cdn.other.com/image2.jpg">
                <img src="/image3.jpg">
            </body>
        </html>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        img_urls = get_image_urls('https://example.com', same_domain=True)

        assert len(img_urls) == 2
        assert all('example.com' in url for url in img_urls)

    @patch('scitex.web._scraping.requests.get')
    def test_get_image_urls_request_failure(self, mock_get):
        """Test handling of request failures."""
        from scitex.web import get_image_urls
        import requests

        mock_get.side_effect = requests.RequestException("Network error")

        img_urls = get_image_urls('https://example.com')

        assert img_urls == []

    @patch('scitex.web._scraping.requests.get')
    def test_get_image_urls_no_images(self, mock_get):
        """Test handling of page with no images."""
        from scitex.web import get_image_urls

        mock_response = Mock()
        mock_response.text = '<html><body>No images here</body></html>'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        img_urls = get_image_urls('https://example.com')

        assert img_urls == []


class TestDownloadImages:
    """Test download_images function."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory after tests."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('scitex.web._scraping.requests.get')
    def test_download_images_basic(self, mock_get):
        """Test basic image downloading."""
        from scitex.web import download_images

        # Mock page response
        page_response = Mock()
        page_response.text = '''
        <html>
            <body>
                <img src="https://example.com/image1.jpg">
                <img src="https://example.com/image2.png">
            </body>
        </html>
        '''
        page_response.raise_for_status = Mock()

        # Mock image responses
        img_response1 = Mock()
        img_response1.content = b'fake image data 1'
        img_response1.headers = {'content-type': 'image/jpeg'}
        img_response1.raise_for_status = Mock()

        img_response2 = Mock()
        img_response2.content = b'fake image data 2'
        img_response2.headers = {'content-type': 'image/png'}
        img_response2.raise_for_status = Mock()

        mock_get.side_effect = [page_response, img_response1, img_response2]

        paths = download_images('https://example.com', output_dir=self.temp_dir)

        assert len(paths) == 2
        assert all(Path(p).exists() for p in paths)

    @patch('scitex.web._scraping.requests.get')
    def test_download_images_with_pattern(self, mock_get):
        """Test image downloading with pattern filter."""
        from scitex.web import download_images

        page_response = Mock()
        page_response.text = '''
        <html>
            <body>
                <img src="https://example.com/image1.jpg">
                <img src="https://example.com/image2.png">
            </body>
        </html>
        '''
        page_response.raise_for_status = Mock()

        img_response = Mock()
        img_response.content = b'fake image data'
        img_response.headers = {'content-type': 'image/jpeg'}
        img_response.raise_for_status = Mock()

        mock_get.side_effect = [page_response, img_response]

        paths = download_images(
            'https://example.com',
            output_dir=self.temp_dir,
            pattern=r'\.jpg$'
        )

        assert len(paths) == 1

    @patch('scitex.web._scraping.requests.get')
    def test_download_images_duplicate_filenames(self, mock_get):
        """Test handling of duplicate filenames."""
        from scitex.web import download_images

        page_response = Mock()
        page_response.text = '''
        <html>
            <body>
                <img src="https://example.com/dir1/image.jpg">
                <img src="https://example.com/dir2/image.jpg">
            </body>
        </html>
        '''
        page_response.raise_for_status = Mock()

        img_response = Mock()
        img_response.content = b'fake image data'
        img_response.headers = {'content-type': 'image/jpeg'}
        img_response.raise_for_status = Mock()

        mock_get.side_effect = [page_response, img_response, img_response]

        paths = download_images('https://example.com', output_dir=self.temp_dir)

        # Should have both images with different filenames
        assert len(paths) == 2
        assert len(set(paths)) == 2  # All paths are unique

    @patch('scitex.web._scraping.requests.get')
    def test_download_images_request_failure(self, mock_get):
        """Test handling of request failures."""
        from scitex.web import download_images
        import requests

        mock_get.side_effect = requests.RequestException("Network error")

        paths = download_images('https://example.com', output_dir=self.temp_dir)

        assert paths == []

    @patch('scitex.web._scraping.requests.get')
    def test_download_images_same_domain(self, mock_get):
        """Test downloading only images from same domain."""
        from scitex.web import download_images

        page_response = Mock()
        page_response.text = '''
        <html>
            <body>
                <img src="https://example.com/image1.jpg">
                <img src="https://cdn.other.com/image2.jpg">
            </body>
        </html>
        '''
        page_response.raise_for_status = Mock()

        img_response = Mock()
        img_response.content = b'fake image data'
        img_response.headers = {'content-type': 'image/jpeg'}
        img_response.raise_for_status = Mock()

        mock_get.side_effect = [page_response, img_response]

        paths = download_images(
            'https://example.com',
            output_dir=self.temp_dir,
            same_domain=True
        )

        # Should only download the first image
        assert len(paths) == 1

    @patch('scitex.web._scraping.requests.get')
    @patch.dict('os.environ', {}, clear=True)
    def test_download_images_no_output_dir(self, mock_get):
        """Test default output directory creation using SCITEX_DIR."""
        from scitex.web import download_images
        import os

        page_response = Mock()
        page_response.text = '''
        <html>
            <body>
                <img src="https://example.com/image.jpg">
            </body>
        </html>
        '''
        page_response.raise_for_status = Mock()

        img_response = Mock()
        img_response.content = b'fake image data'
        img_response.headers = {'content-type': 'image/jpeg'}
        img_response.raise_for_status = Mock()

        mock_get.side_effect = [page_response, img_response]

        # Set SCITEX_DIR to a temp location for testing
        test_scitex_dir = Path(self.temp_dir) / 'scitex'
        os.environ['SCITEX_DIR'] = str(test_scitex_dir)

        paths = download_images('https://example.com')

        assert len(paths) == 1
        expected_dir = test_scitex_dir / 'web' / 'downloads'
        assert expected_dir.exists()

    @patch('scitex.web._scraping.requests.get')
    @patch.dict('os.environ', {'SCITEX_WEB_DOWNLOADS_DIR': '/tmp/test_downloads'}, clear=True)
    def test_download_images_env_var_priority(self, mock_get):
        """Test that SCITEX_WEB_DOWNLOADS_DIR takes priority."""
        from scitex.web import download_images
        import os

        page_response = Mock()
        page_response.text = '''
        <html>
            <body>
                <img src="https://example.com/image.jpg">
            </body>
        </html>
        '''
        page_response.raise_for_status = Mock()

        img_response = Mock()
        img_response.content = b'fake image data'
        img_response.headers = {'content-type': 'image/jpeg'}
        img_response.raise_for_status = Mock()

        mock_get.side_effect = [page_response, img_response]

        # Set both env vars
        os.environ['SCITEX_DIR'] = '/tmp/scitex'
        os.environ['SCITEX_WEB_DOWNLOADS_DIR'] = self.temp_dir

        paths = download_images('https://example.com')

        # Should use SCITEX_WEB_DOWNLOADS_DIR, not SCITEX_DIR
        assert len(paths) == 1
        assert paths[0].startswith(self.temp_dir)

    @patch('scitex.web._scraping.requests.get')
    @patch('scitex.web._scraping.PILLOW_AVAILABLE', True)
    @patch('scitex.web._scraping.Image.open')
    def test_download_images_min_size_filter(self, mock_image_open, mock_get):
        """Test minimum size filtering."""
        from scitex.web import download_images

        page_response = Mock()
        page_response.text = '''
        <html>
            <body>
                <img src="https://example.com/small.jpg">
                <img src="https://example.com/large.jpg">
            </body>
        </html>
        '''
        page_response.raise_for_status = Mock()

        img_response_small = Mock()
        img_response_small.content = b'small image'
        img_response_small.headers = {'content-type': 'image/jpeg'}
        img_response_small.raise_for_status = Mock()

        img_response_large = Mock()
        img_response_large.content = b'large image'
        img_response_large.headers = {'content-type': 'image/jpeg'}
        img_response_large.raise_for_status = Mock()

        # Mock image sizes
        small_img = Mock()
        small_img.size = (50, 50)
        large_img = Mock()
        large_img.size = (500, 500)

        mock_image_open.side_effect = [small_img, large_img]
        mock_get.side_effect = [
            page_response,
            img_response_small,
            img_response_large
        ]

        paths = download_images(
            'https://example.com',
            output_dir=self.temp_dir,
            min_size=(100, 100)
        )

        # Only the large image should be downloaded
        assert len(paths) == 1


class TestScrapingModuleImport:
    """Test that scraping functions are properly exported."""

    def test_scraping_functions_available(self):
        """Test that all scraping functions are available."""
        import scitex.web

        assert hasattr(scitex.web, 'get_urls')
        assert hasattr(scitex.web, 'download_images')
        assert hasattr(scitex.web, 'get_image_urls')

        assert callable(scitex.web.get_urls)
        assert callable(scitex.web.download_images)
        assert callable(scitex.web.get_image_urls)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/web/_scraping.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: ./src/scitex/web/_scraping.py
# 
# """Web scraping utilities for extracting URLs and downloading images."""
# 
# import os
# import re
# import urllib.parse
# from datetime import datetime
# from pathlib import Path
# from typing import List, Optional, Set, Tuple
# from concurrent.futures import ThreadPoolExecutor, as_completed
# 
# import requests
# from bs4 import BeautifulSoup
# from tqdm import tqdm
# 
# try:
#     from PIL import Image
#     from io import BytesIO
# 
#     PILLOW_AVAILABLE = True
# except ImportError:
#     PILLOW_AVAILABLE = False
# 
# from scitex.logging import getLogger
# 
# logger = getLogger(__name__)
# 
# 
# def _get_default_download_dir() -> str:
#     """Get default download directory using SCITEX_DIR if available."""
#     scitex_root = os.environ.get("SCITEX_DIR")
#     if scitex_root is None:
#         scitex_root = os.path.expanduser("~/.scitex")
#     return os.path.join(scitex_root, "web", "downloads")
# 
# 
# def get_urls(
#     url: str,
#     pattern: Optional[str] = None,
#     absolute: bool = True,
#     same_domain: bool = False,
#     include_external: bool = True,
# ) -> List[str]:
#     """
#     Extract all URLs from a webpage.
# 
#     Args:
#         url: The URL of the webpage to scrape
#         pattern: Optional regex pattern to filter URLs (e.g., r'\.pdf$' for PDF files)
#         absolute: If True, convert relative URLs to absolute URLs
#         same_domain: If True, only return URLs from the same domain
#         include_external: If True, include external links (only applies if same_domain=False)
# 
#     Returns:
#         List of URLs found on the page
# 
#     Example:
#         >>> urls = get_urls('https://example.com', pattern=r'\.pdf$')
#         >>> urls = get_urls('https://example.com', same_domain=True)
#     """
#     try:
#         logger.info(f"Fetching URLs from: {url}")
#         response = requests.get(url, timeout=30)
#         response.raise_for_status()
#     except requests.RequestException as e:
#         logger.error(f"Failed to fetch URL {url}: {e}")
#         return []
# 
#     soup = BeautifulSoup(response.text, "html.parser")
#     urls_found: Set[str] = set()
# 
#     # Parse the base domain
#     parsed_base = urllib.parse.urlparse(url)
#     base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
# 
#     # Find all links
#     for link in soup.find_all("a", href=True):
#         href = link["href"]
# 
#         # Convert to absolute URL if requested
#         if absolute:
#             href = urllib.parse.urljoin(url, href)
# 
#         # Filter by domain if requested
#         if same_domain:
#             parsed_href = urllib.parse.urlparse(href)
#             if parsed_href.netloc != parsed_base.netloc:
#                 continue
#         elif not include_external:
#             parsed_href = urllib.parse.urlparse(href)
#             if parsed_href.netloc and parsed_href.netloc != parsed_base.netloc:
#                 continue
# 
#         # Filter by pattern if provided
#         if pattern:
#             if not re.search(pattern, href):
#                 continue
# 
#         urls_found.add(href)
# 
#     result = sorted(list(urls_found))
#     logger.info(f"Found {len(result)} URLs")
#     return result
# 
# 
# def download_images(
#     url: str,
#     output_dir: Optional[str] = None,
#     pattern: Optional[str] = None,
#     min_size: Optional[Tuple[int, int]] = None,
#     max_workers: int = 5,
#     same_domain: bool = False,
# ) -> List[str]:
#     """
#     Download all images from a webpage.
# 
#     Args:
#         url: The URL of the webpage to scrape
#         output_dir: Directory to save images. Priority:
#                    1. This parameter if specified
#                    2. $SCITEX_WEB_DOWNLOADS_DIR environment variable
#                    3. $SCITEX_DIR/web/downloads (default)
#         pattern: Optional regex pattern to filter image URLs
#         min_size: Optional minimum size as (width, height) tuple to filter images
#         max_workers: Number of concurrent download threads
#         same_domain: If True, only download images from the same domain
# 
#     Returns:
#         List of paths to downloaded images
# 
#     Note:
#         - SVG files are automatically skipped (vector graphics)
#         - Images are saved in timestamped subdirectories: images-YYYYMMDD_HHMMSS/
# 
#     Example:
#         >>> paths = download_images('https://example.com', output_dir='./downloads')
#         >>> paths = download_images('https://example.com', min_size=(100, 100))
#         >>> # Uses $SCITEX_WEB_DOWNLOADS_DIR or $SCITEX_DIR/web/downloads
#         >>> paths = download_images('https://example.com')
#     """
#     if not PILLOW_AVAILABLE:
#         logger.warning("Pillow is not available. min_size filtering will be disabled.")
# 
#     # Set default output directory
#     if output_dir is None:
#         # Check SCITEX_WEB_DOWNLOADS_DIR first
#         output_dir = os.environ.get("SCITEX_WEB_DOWNLOADS_DIR")
#         if output_dir is None:
#             # Fall back to SCITEX_DIR/web/downloads
#             output_dir = _get_default_download_dir()
# 
#     # Create timestamped subdirectory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_path = Path(output_dir).expanduser() / f"images-{timestamp}"
#     output_path.mkdir(parents=True, exist_ok=True)
# 
#     logger.info(f"Saving images to: {output_path}")
# 
#     try:
#         logger.info(f"Fetching page: {url}")
#         response = requests.get(url, timeout=30)
#         response.raise_for_status()
#     except requests.RequestException as e:
#         logger.error(f"Failed to fetch URL {url}: {e}")
#         return []
# 
#     soup = BeautifulSoup(response.text, "html.parser")
#     image_urls: Set[str] = set()
# 
#     # Parse the base domain
#     parsed_base = urllib.parse.urlparse(url)
# 
#     # Find all image tags
#     for img in soup.find_all("img", src=True):
#         img_url = img["src"]
# 
#         # Convert to absolute URL
#         img_url = urllib.parse.urljoin(url, img_url)
# 
#         # Skip SVG files (vector graphics, not raster images)
#         if img_url.lower().endswith((".svg", ".svgz")):
#             continue
# 
#         # Filter by domain if requested
#         if same_domain:
#             parsed_img = urllib.parse.urlparse(img_url)
#             if parsed_img.netloc != parsed_base.netloc:
#                 continue
# 
#         # Filter by pattern if provided
#         if pattern:
#             if not re.search(pattern, img_url):
#                 continue
# 
#         image_urls.add(img_url)
# 
#     logger.info(f"Found {len(image_urls)} images")
# 
#     # Download images
#     downloaded_paths = []
# 
#     def download_image(img_url: str) -> Optional[str]:
#         try:
#             img_response = requests.get(img_url, timeout=30)
#             img_response.raise_for_status()
# 
#             # Check image size if requested and Pillow is available
#             if min_size and PILLOW_AVAILABLE:
#                 try:
#                     img = Image.open(BytesIO(img_response.content))
#                     if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
#                         return None
#                 except Exception:
#                     pass
# 
#             # Generate filename from URL
#             parsed_url = urllib.parse.urlparse(img_url)
#             filename = Path(parsed_url.path).name
# 
#             # If filename is empty or doesn't have extension, generate one
#             if not filename or "." not in filename:
#                 ext = ".jpg"  # default extension
#                 if "content-type" in img_response.headers:
#                     content_type = img_response.headers["content-type"]
#                     if "png" in content_type:
#                         ext = ".png"
#                     elif "gif" in content_type:
#                         ext = ".gif"
#                     elif "webp" in content_type:
#                         ext = ".webp"
#                 filename = f"image_{hash(img_url)}{ext}"
# 
#             # Save image
#             file_path = output_path / filename
# 
#             # Handle duplicate filenames
#             counter = 1
#             original_stem = file_path.stem
#             while file_path.exists():
#                 file_path = output_path / f"{original_stem}_{counter}{file_path.suffix}"
#                 counter += 1
# 
#             with open(file_path, "wb") as f:
#                 f.write(img_response.content)
# 
#             return str(file_path)
# 
#         except Exception as e:
#             logger.warning(f"Failed to download image {img_url}: {e}")
#             return None
# 
#     # Download images concurrently
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_url = {
#             executor.submit(download_image, img_url): img_url for img_url in image_urls
#         }
# 
#         for future in tqdm(
#             as_completed(future_to_url),
#             total=len(image_urls),
#             desc="Downloading images",
#         ):
#             result = future.result()
#             if result:
#                 downloaded_paths.append(result)
# 
#     logger.info(f"Downloaded {len(downloaded_paths)} images to {output_dir}")
#     return downloaded_paths
# 
# 
# def get_image_urls(
#     url: str,
#     pattern: Optional[str] = None,
#     same_domain: bool = False,
# ) -> List[str]:
#     """
#     Extract all image URLs from a webpage without downloading them.
# 
#     Args:
#         url: The URL of the webpage to scrape
#         pattern: Optional regex pattern to filter image URLs
#         same_domain: If True, only return images from the same domain
# 
#     Returns:
#         List of image URLs found on the page
# 
#     Note:
#         - SVG files are automatically skipped (vector graphics)
# 
#     Example:
#         >>> img_urls = get_image_urls('https://example.com')
#         >>> img_urls = get_image_urls('https://example.com', pattern=r'\.png$')
#     """
#     try:
#         logger.info(f"Fetching image URLs from: {url}")
#         response = requests.get(url, timeout=30)
#         response.raise_for_status()
#     except requests.RequestException as e:
#         logger.error(f"Failed to fetch URL {url}: {e}")
#         return []
# 
#     soup = BeautifulSoup(response.text, "html.parser")
#     image_urls: Set[str] = set()
# 
#     # Parse the base domain
#     parsed_base = urllib.parse.urlparse(url)
# 
#     # Find all image tags
#     for img in soup.find_all("img", src=True):
#         img_url = img["src"]
# 
#         # Convert to absolute URL
#         img_url = urllib.parse.urljoin(url, img_url)
# 
#         # Skip SVG files (vector graphics, not raster images)
#         if img_url.lower().endswith((".svg", ".svgz")):
#             continue
# 
#         # Filter by domain if requested
#         if same_domain:
#             parsed_img = urllib.parse.urlparse(img_url)
#             if parsed_img.netloc != parsed_base.netloc:
#                 continue
# 
#         # Filter by pattern if provided
#         if pattern:
#             if not re.search(pattern, img_url):
#                 continue
# 
#         image_urls.add(img_url)
# 
#     result = sorted(list(image_urls))
#     logger.info(f"Found {len(result)} image URLs")
#     return result
# 
# 
# if __name__ == "__main__":
#     import argparse
# 
#     parser = argparse.ArgumentParser(description="Web scraping utilities")
#     parser.add_argument("url", type=str, help="URL to scrape")
#     parser.add_argument(
#         "--mode",
#         "-m",
#         choices=["urls", "images", "image_urls"],
#         default="urls",
#         help="Scraping mode",
#     )
#     parser.add_argument("--output", "-o", type=str, help="Output directory for images")
#     parser.add_argument(
#         "--pattern", "-p", type=str, help="Regex pattern to filter URLs"
#     )
#     parser.add_argument(
#         "--same-domain", action="store_true", help="Only include URLs from same domain"
#     )
#     parser.add_argument(
#         "--min-size", type=str, help="Minimum image size as WIDTHxHEIGHT"
#     )
# 
#     args = parser.parse_args()
# 
#     if args.mode == "urls":
#         urls = get_urls(args.url, pattern=args.pattern, same_domain=args.same_domain)
#         for url in urls:
#             print(url)
#     elif args.mode == "images":
#         min_size = None
#         if args.min_size:
#             width, height = map(int, args.min_size.split("x"))
#             min_size = (width, height)
# 
#         paths = download_images(
#             args.url,
#             output_dir=args.output,
#             pattern=args.pattern,
#             min_size=min_size,
#             same_domain=args.same_domain,
#         )
#         for path in paths:
#             print(path)
#     elif args.mode == "image_urls":
#         img_urls = get_image_urls(
#             args.url, pattern=args.pattern, same_domain=args.same_domain
#         )
#         for img_url in img_urls:
#             print(img_url)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/web/_scraping.py
# --------------------------------------------------------------------------------
