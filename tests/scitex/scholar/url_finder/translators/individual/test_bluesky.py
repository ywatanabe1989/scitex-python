# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/bluesky.py
# --------------------------------------------------------------------------------
# """
# Bluesky Translator
# 
# Translates posts from Bluesky social network.
# 
# Metadata:
#     translatorID: 3bba003a-ad42-457e-9ea1-547df39d9d00
#     label: Bluesky
#     creator: Stephan Hügel
#     target: ^https://bsky\.app/
#     minVersion: 5.0
#     priority: 100
#     inRepository: True
#     translatorType: 4
#     browserSupport: gcsibv
#     lastUpdated: 2025-03-26 14:26:25
# """
# 
# from typing import Dict, Any, List, Optional
# from bs4 import BeautifulSoup
# import re
# 
# 
# class BlueskyTranslator:
#     """Translator for Bluesky posts."""
# 
#     METADATA = {
#         "translatorID": "3bba003a-ad42-457e-9ea1-547df39d9d00",
#         "label": "Bluesky",
#         "creator": "Stephan Hügel",
#         "target": r"^https://bsky\.app/",
#         "minVersion": "5.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 4,
#         "browserSupport": "gcsibv",
#         "lastUpdated": "2025-03-26 14:26:25",
#     }
# 
#     HANDLE_RE = re.compile(r"(?:/profile/)(([^/]+))")
#     POST_ID_RE = re.compile(r"(?:/post/)([a-zA-Z0-9]+)")
# 
#     def detect_web(self, doc: BeautifulSoup, url: str) -> str:
#         """Detect page type."""
#         if (
#             "/post/" in url
#             and self.HANDLE_RE.search(url)
#             and self.POST_ID_RE.search(url)
#         ):
#             return "forumPost"
#         return ""
# 
#     def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
#         """Extract data from the page using API."""
#         return [self.scrape_api(doc, url)]
# 
#     def scrape_api(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
#         """
#         Scrape post data using Bluesky public API.
# 
#         Args:
#             doc: BeautifulSoup parsed document (for snapshot)
#             url: URL of the post
# 
#         Returns:
#             Dictionary containing post metadata and API request info
#         """
#         # Extract handle and post ID from URL
#         handle_match = self.HANDLE_RE.search(url)
#         post_id_match = self.POST_ID_RE.search(url)
# 
#         if not handle_match or not post_id_match:
#             return {"error": "Could not extract handle or post ID from URL"}
# 
#         found_handle = handle_match.group(1)
#         found_post_id = post_id_match.group(1)
# 
#         # Construct API URL
#         api_url = (
#             f"https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread"
#             f"?uri=at://{found_handle}/app.bsky.feed.post/{found_post_id}"
#         )
# 
#         item = {
#             "itemType": "forumPost",
#             "forumTitle": "Bluesky",
#             "postType": "Post",
#             "url": url,
#             "creators": [],
#             "tags": [],
#             "notes": [],
#             "attachments": [],
#             "_api_url": api_url,  # Signal that API request is needed
#         }
# 
#         # Note: In a full implementation, this would make the API request
#         # and populate the fields. For this translation, we provide the
#         # structure and API URL for the implementation layer to use.
# 
#         # The API response structure would include:
#         # - post.record.text (for title/abstract)
#         # - post.record.createdAt (for date)
#         # - post.author.displayName or post.author.handle (for creator)
#         # - post.author.did (for DID extra field)
#         # - post.likeCount, repostCount, quoteCount (for extra fields)
#         # - post.embed.record (for quoted posts - adds note)
#         # - thread.replies (for reply count - adds note)
# 
#         # Add snapshot
#         item["attachments"].append(
#             {"title": "Snapshot", "mimeType": "text/html", "url": url}
#         )
# 
#         return item
# 
#     @staticmethod
#     def _ellipsize(text: str, max_length: int, word_boundary: bool = True) -> str:
#         """Truncate text to max_length with ellipsis."""
#         if len(text) <= max_length:
#             return text
# 
#         if word_boundary:
#             # Find last space before max_length
#             truncated = text[:max_length]
#             last_space = truncated.rfind(" ")
#             if last_space > 0:
#                 return truncated[:last_space] + "..."
# 
#         return text[:max_length] + "..."

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/bluesky.py
# --------------------------------------------------------------------------------
