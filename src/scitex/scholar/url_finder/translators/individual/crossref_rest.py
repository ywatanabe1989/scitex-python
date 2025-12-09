#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python implementation of Crossref REST API Zotero translator.

Original JavaScript: Crossref REST.js
Translator ID: 0a61e167-de9a-4f93-a68a-628b48855909

This translator uses the Crossref REST API to fetch metadata by DOI.
API documentation:
- https://github.com/Crossref/rest-api-doc
- https://github.com/Crossref/rest-api-doc/blob/master/api_format.md
- http://api.crossref.org/types

Note: This is a search translator (translatorType 8) that doesn't match URLs
but is called programmatically with DOI or query parameters.
"""

import re
import html
import json
from typing import List, Dict, Any, Optional
from urllib.parse import quote

from playwright.async_api import Page
from ..core.base import BaseTranslator


class CrossrefRESTTranslator(BaseTranslator):
    """Crossref REST API metadata fetcher - Python implementation."""

    LABEL = "Crossref REST"
    # This is a search translator - no URL pattern matching
    URL_TARGET_PATTERN = r""
    TRANSLATOR_TYPE = 8  # Search translator

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Crossref REST is a search translator and doesn't match URLs.

        Args:
            url: URL to check

        Returns:
            Always False - this translator is invoked programmatically
        """
        return False

    @classmethod
    def remove_unsupported_markup(cls, text: str) -> str:
        """Remove unsupported HTML markup while preserving allowed tags.

        Allowed tags: i, b, sub, sup, span, sc
        Transform tags: scp -> span with small-caps style

        Args:
            text: Text with HTML markup

        Returns:
            Text with only supported markup
        """
        if not text:
            return text

        # Remove CDATA markup
        text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.DOTALL)

        # Define allowed and transform patterns
        supported = ["i", "b", "sub", "sup", "span", "sc"]
        transform = {"scp": '<span style="font-variant:small-caps;">'}

        def replace_markup(match):
            closing = match.group(1) or ""
            tag_name = match.group(2).lower()

            if tag_name in supported:
                return f"<{closing}{tag_name}>"

            if tag_name in transform:
                return "</span>" if closing else transform[tag_name]

            return ""

        text = re.sub(r"<(/?)(\w+)[^<>]*>", replace_markup, text)
        return text

    @classmethod
    def decode_entities(cls, text: str) -> str:
        """Decode HTML entities and remove newlines.

        Args:
            text: Text with HTML entities

        Returns:
            Decoded text
        """
        if not text:
            return text

        # Remove newlines
        text = text.replace("\n", "")

        # Decode common entities
        entities = {"&amp;": "&", "&quot;": '"', "&lt;": "<", "&gt;": ">"}

        for entity, char in entities.items():
            text = text.replace(entity, char)

        return text

    @classmethod
    def fix_author_capitalization(cls, name: str) -> Optional[str]:
        """Fix all-uppercase author names to title case.

        Args:
            name: Author name (may be all uppercase)

        Returns:
            Name with proper capitalization, or None if input is None
        """
        if not name:
            return name

        if isinstance(name, str) and name.isupper():
            # Convert to title case
            return name.title()

        return name

    @classmethod
    def parse_creators(
        cls,
        result: Dict[str, Any],
        item: Dict[str, Any],
        type_override_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Parse creators from Crossref result and add to item.

        Args:
            result: Crossref API result
            item: Item dictionary to populate
            type_override_map: Optional mapping of creator types
        """
        if type_override_map is None:
            type_override_map = {}

        if "creators" not in item:
            item["creators"] = []

        types = ["author", "editor", "chair", "translator"]

        for creator_type in types:
            if creator_type not in result:
                continue

            # Determine Zotero creator type
            if creator_type in type_override_map:
                zotero_type = type_override_map[creator_type]
            elif creator_type in ["author", "editor", "translator"]:
                zotero_type = creator_type
            else:
                zotero_type = "contributor"

            if not zotero_type:
                continue

            for creator in result[creator_type]:
                new_creator = {"creatorType": zotero_type}

                if "name" in creator and creator["name"]:
                    # Institutional author
                    new_creator["fieldMode"] = 1
                    new_creator["lastName"] = creator["name"]
                else:
                    # Individual author
                    given = creator.get("given", "")
                    family = creator.get("family", "")

                    new_creator["firstName"] = cls.fix_author_capitalization(given)
                    new_creator["lastName"] = cls.fix_author_capitalization(family)

                    if not new_creator["firstName"]:
                        new_creator["fieldMode"] = 1

                item["creators"].append(new_creator)

    @classmethod
    def parse_date(cls, date_obj: Optional[Dict[str, Any]]) -> Optional[str]:
        """Parse Crossref date object to Zotero date format.

        Args:
            date_obj: Crossref date object with 'date-parts' array

        Returns:
            Formatted date string or None
        """
        if not date_obj or "date-parts" not in date_obj:
            return None

        if not date_obj["date-parts"] or not date_obj["date-parts"][0]:
            return None

        parts = date_obj["date-parts"][0]
        year = parts[0] if len(parts) > 0 else None
        month = parts[1] if len(parts) > 1 else None
        day = parts[2] if len(parts) > 2 else None

        if year:
            if month:
                if day:
                    return f"{year}-{month:02d}-{day:02d}"
                else:
                    return f"{month:02d}/{year}"
            else:
                return str(year)

        return None

    @classmethod
    def process_crossref_result(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single Crossref API result into Zotero item format.

        Args:
            result: Single item from Crossref API response

        Returns:
            Zotero item dictionary
        """
        item = {}
        creator_type_override_map = {}

        # Get institution early to avoid UnboundLocalError
        institution = result.get("institution")

        # Determine item type
        result_type = result.get("type", "")

        if result_type in [
            "journal",
            "journal-article",
            "journal-volume",
            "journal-issue",
        ]:
            item["itemType"] = "journalArticle"
        elif result_type in ["report", "report-series", "report-component"]:
            item["itemType"] = "report"
        elif result_type in [
            "book",
            "book-series",
            "book-set",
            "book-track",
            "monograph",
            "reference-book",
            "edited-book",
        ]:
            item["itemType"] = "book"
        elif result_type in [
            "book-chapter",
            "book-part",
            "book-section",
            "reference-entry",
        ]:
            item["itemType"] = "bookSection"
            creator_type_override_map = {"author": "bookAuthor"}
        elif (
            result_type == "other"
            and result.get("ISBN")
            and result.get("container-title")
        ):
            item["itemType"] = "bookSection"
            container = result["container-title"]
            if len(container) >= 2:
                item["seriesTitle"] = container[0]
                item["bookTitle"] = container[1]
            else:
                item["bookTitle"] = container[0]
            creator_type_override_map = {"author": "bookAuthor"}
        elif result_type == "standard":
            item["itemType"] = "document"  # Using document for standards
        elif result_type in ["dataset", "database"]:
            item["itemType"] = "document"  # Using document for datasets
        elif result_type in [
            "proceedings",
            "proceedings-article",
            "proceedings-series",
        ]:
            item["itemType"] = "conferencePaper"
        elif result_type == "dissertation":
            item["itemType"] = "thesis"
            item["date"] = cls.parse_date(result.get("approved"))
            degree = result.get("degree")
            if degree and len(degree) > 0:
                item["thesisType"] = re.sub(r"\(.+\)", "", degree[0])
        elif result_type == "posted-content":
            if result.get("subtype") == "preprint":
                item["itemType"] = "document"  # Using document for preprints
                item["archive"] = result.get("group-title")
            else:
                item["itemType"] = "blogPost"
                if institution and len(institution) > 0 and institution[0].get("name"):
                    item["blogTitle"] = institution[0]["name"]
        elif result_type == "peer-review":
            item["itemType"] = "document"
            item["type"] = "peer review"
            if not result.get("author"):
                item["creators"] = [
                    {
                        "lastName": "Anonymous Reviewer",
                        "fieldMode": 1,
                        "creatorType": "author",
                    }
                ]

            # Add review relationship as note
            relation = result.get("relation", {}).get("is-review-of", [])
            if relation and len(relation) > 0:
                review_of = relation[0]
                id_type = review_of.get("id-type")
                review_id = review_of.get("id")

                if id_type == "doi":
                    identifier = f'<a href="https://doi.org/{review_id}">https://doi.org/{review_id}</a>'
                elif id_type == "url":
                    identifier = f'<a href="{review_id}">{review_id}</a>'
                else:
                    identifier = review_id

                if "notes" not in item:
                    item["notes"] = []
                item["notes"].append(f"Review of {identifier}")
        else:
            item["itemType"] = "document"

        # Parse creators
        cls.parse_creators(result, item, creator_type_override_map)

        # Add description as note
        if result.get("description"):
            if "notes" not in item:
                item["notes"] = []
            item["notes"].append(result["description"])

        # Parse fields
        abstract = result.get("abstract")
        if abstract:
            item["abstractNote"] = cls.remove_unsupported_markup(abstract)

        # Page/article number
        item["pages"] = result.get("page") or result.get("article-number")

        # Identifiers
        isbn_list = result.get("ISBN")
        if isbn_list:
            item["ISBN"] = ", ".join(isbn_list)

        issn_list = result.get("ISSN")
        if issn_list:
            item["ISSN"] = ", ".join(issn_list)

        item["issue"] = result.get("issue")
        item["volume"] = result.get("volume")
        item["language"] = result.get("language")
        item["edition"] = result.get("edition-number")

        # Publisher
        publisher = result.get("publisher")
        item["publisher"] = publisher
        if item["itemType"] == "thesis":
            item["university"] = publisher
        # For reports, set institution
        if item["itemType"] == "report":
            item["institution"] = publisher

        # Container title
        container_title = result.get("container-title")
        if container_title and len(container_title) > 0:
            if item["itemType"] == "journalArticle":
                item["publicationTitle"] = container_title[0]
            elif item["itemType"] == "conferencePaper":
                item["proceedingsTitle"] = container_title[0]
            elif item["itemType"] == "book":
                item["series"] = container_title[0]
            elif item["itemType"] == "bookSection" and "bookTitle" not in item:
                item["bookTitle"] = container_title[0]
            else:
                item["seriesTitle"] = container_title[0]

        # Conference name
        event = result.get("event")
        if event:
            item["conferenceName"] = event.get("name")

        # Short container title (journal abbreviation)
        short_container = result.get("short-container-title")
        if short_container and len(short_container) > 0:
            if container_title and short_container[0] != container_title[0]:
                item["journalAbbreviation"] = short_container[0]

        # Place
        if event and event.get("location"):
            item["place"] = event["location"]
        elif institution and len(institution) > 0 and institution[0].get("place"):
            places = institution[0]["place"]
            if isinstance(places, list):
                item["place"] = ", ".join(places)
            else:
                item["place"] = places
        else:
            item["place"] = result.get("publisher-location")

        # Institution/University
        if institution and len(institution) > 0 and institution[0].get("name"):
            if item["itemType"] == "thesis":
                item["university"] = institution[0]["name"]
            else:
                item["institution"] = institution[0]["name"]

        # Date - prefer print to other dates
        published_print = cls.parse_date(result.get("published-print"))
        if published_print:
            item["date"] = published_print
        else:
            issued = cls.parse_date(result.get("issued"))
            if issued:
                item["date"] = issued

        # DOI
        item["DOI"] = result.get("DOI")

        # URL
        resource = result.get("resource", {}).get("primary", {})
        if resource.get("URL"):
            item["url"] = resource["URL"]

        # Rights/License
        license_list = result.get("license")
        if license_list and len(license_list) > 0 and license_list[0].get("URL"):
            item["rights"] = license_list[0]["URL"]

        # Title
        title_list = result.get("title")
        if title_list and len(title_list) > 0:
            item["title"] = title_list[0]

            # Add subtitle
            subtitle_list = result.get("subtitle")
            if subtitle_list and len(subtitle_list) > 0:
                subtitle = subtitle_list[0]
                # Avoid duplicating subtitle if already in title
                if subtitle.lower() not in item["title"].lower():
                    # Add colon if not already present
                    if not item["title"].endswith(":"):
                        item["title"] += ":"
                    item["title"] += f" {subtitle}"

            item["title"] = cls.remove_unsupported_markup(item["title"])

        if not item.get("title"):
            item["title"] = "[No title found]"

        # Fix character encoding issues
        for field, value in item.items():
            if isinstance(value, str):
                # Check for control characters
                if re.search(r"[\u007F-\u009F]", value):
                    try:
                        # Try to fix double-encoding issues
                        value = html.unescape(value)
                    except:
                        # Strip control characters
                        value = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", value)

                item[field] = cls.decode_entities(value)

        item["libraryCatalog"] = "Crossref"

        return item

    @classmethod
    async def search_by_doi(
        cls, doi: str, mailto: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search Crossref by DOI.

        Args:
            doi: DOI identifier (can be string or list)
            mailto: Optional email for polite pool access

        Returns:
            List of Zotero items
        """
        import aiohttp

        # Handle list of DOIs
        if isinstance(doi, list):
            filter_param = "doi:" + ",doi:".join([d.strip() for d in doi if d])
        else:
            filter_param = f"doi:{doi.strip()}"

        query = f"?filter={quote(filter_param)}"

        if mailto:
            query += f"&mailto={quote(mailto)}"

        url = f"https://api.crossref.org/works/{query}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                items = []

                if "message" in data and "items" in data["message"]:
                    for result in data["message"]["items"]:
                        item = cls.process_crossref_result(result)
                        items.append(item)

                return items

    @classmethod
    async def search_by_query(
        cls, query: str, mailto: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search Crossref by bibliographic query.

        Args:
            query: Bibliographic search query
            mailto: Optional email for polite pool access

        Returns:
            List of Zotero items
        """
        import aiohttp

        query_param = f"?query.bibliographic={quote(query)}"

        if mailto:
            query_param += f"&mailto={quote(mailto)}"

        url = f"https://api.crossref.org/works/{query_param}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                items = []

                if "message" in data and "items" in data["message"]:
                    for result in data["message"]["items"]:
                        item = cls.process_crossref_result(result)
                        items.append(item)

                return items

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Not applicable for search translator.

        Args:
            page: Playwright page (not used)

        Returns:
            Empty list
        """
        return []


if __name__ == "__main__":
    import asyncio

    async def main():
        """Demonstration of CrossrefRESTTranslator usage."""
        print("=" * 70)
        print("Testing CrossrefRESTTranslator")
        print("=" * 70)

        # Test DOI search
        test_doi = "10.1038/s41467-020-15908-3"
        print(f"\nSearching by DOI: {test_doi}")

        items = await CrossrefRESTTranslator.search_by_doi(test_doi)

        if items:
            item = items[0]
            print(f"\nResults:")
            print(f"  Type: {item.get('itemType')}")
            print(f"  Title: {item.get('title')}")
            print(f"  DOI: {item.get('DOI')}")
            print(f"  Journal: {item.get('publicationTitle')}")
            print(f"  Date: {item.get('date')}")
            print(f"  Authors: {len(item.get('creators', []))}")
            if item.get("creators"):
                print(f"    First: {item['creators'][0]}")
        else:
            print("No results found")

        # Test query search
        print(f"\n{'=' * 70}")
        test_query = "machine learning neural networks"
        print(f"Searching by query: {test_query}")

        items = await CrossrefRESTTranslator.search_by_query(test_query)
        print(f"\nFound {len(items)} results")

        if items:
            print(f"\nFirst result:")
            item = items[0]
            print(f"  Type: {item.get('itemType')}")
            print(f"  Title: {item.get('title')}")
            print(f"  DOI: {item.get('DOI')}")

    asyncio.run(main())


# EOF
