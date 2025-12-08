"""
Dataverse Translator

Translates Dataverse dataset pages to Zotero format.

Metadata:
    translatorID: aedf3fb0-9a50-47b3-ba2f-3206552b82a9
    label: Dataverse
    creator: Abe Jellinek, Sebastian Karcher
    target: ^https?://(www\\.)?((open|research-?|hei|planetary-|osna|in|bonn|borealis|lida\\.|archaeology\\.|entrepot\\.recherche\\.|archive\\.|redape\\.)?(data|e?da[td]os)|dvn|sodha\\.be|repositorio(\\.|dedados|pesquisas)|abacus\\.library\\.ubc\\.ca|dorel\\.univ-lorraine\\.fr|darus\\.uni-stuttgart\\.de|dunas\\.ua\\.pt|edmond\\.mpdl\\.mpg\\.de|keen\\.zih\\.tu-dresden\\.de|rdr\\.kuleuven\\.be|portal\\.odissei\\.nl)
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2023-05-01 12:09:04
"""

from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
import json
import re


class DataverseTranslator:
    """Translator for Dataverse research data repositories."""

    METADATA = {
        "translatorID": "aedf3fb0-9a50-47b3-ba2f-3206552b82a9",
        "label": "Dataverse",
        "creator": "Abe Jellinek, Sebastian Karcher",
        "target": r"^https?://(www\.)?((open|research-?|hei|planetary-|osna|in|bonn|borealis|lida\.|archaeology\.|entrepot\.recherche\.|archive\.|redape\.)?(data|e?da[td]os)|dvn|sodha\.be|repositorio(\.|dedados|pesquisas)|abacus\.library\.ubc\.ca|dorel\.univ-lorraine\.fr|darus\.uni-stuttgart\.de|dunas\.ua\.pt|edmond\.mpdl\.mpg\.de|keen\.zih\.tu-dresden\.de|rdr\.kuleuven\.be|portal\.odissei\.nl)",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2023-05-01 12:09:04",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is Dataverse dataset or search."""
        if not doc.select_one("#dataverseHeader"):
            return ""

        if "/dataset.xhtml" in url:
            return "dataset"
        elif self.get_search_results(doc, check_only=True):
            return "multiple"

        return ""

    def get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Optional[Dict[str, str]]:
        """Extract search results."""
        items = {}
        rows = doc.select('.datasetResult a[href*="/dataset.xhtml"]')

        for row in rows:
            href = row.get("href")
            title = row.get_text(strip=True)

            if not href or not title:
                continue

            if check_only:
                return {"found": "true"}

            items[href] = title

        return items if items else None

    def do_web(self, doc: BeautifulSoup, url: str) -> Any:
        """Main extraction method."""
        page_type = self.detect_web(doc, url)

        if page_type == "multiple":
            return self.get_search_results(doc, check_only=False)
        else:
            return self.scrape(doc, url)

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Scrape dataset metadata."""
        item = {
            "itemType": "dataset",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Parse JSON-LD schema
        ld_json_tag = doc.select_one('script[type="application/ld+json"]')
        if ld_json_tag and ld_json_tag.string:
            try:
                schema = json.loads(ld_json_tag.string)

                # Extract license
                license_info = schema.get("license")
                if isinstance(license_info, dict):
                    license_info = license_info.get("text", "")
                if license_info:
                    # Clean HTML tags from license text
                    item["rights"] = re.sub(r"<[^>]+>", "", str(license_info)).strip()

                # Extract abstract
                if schema.get("description"):
                    item["abstractNote"] = schema["description"]

                # Extract version
                version = schema.get("version")
                if version and version > 1:
                    item["versionNumber"] = str(version)

            except (json.JSONDecodeError, KeyError):
                pass

        # Extract library catalog (repository name)
        breadcrumb = doc.select_one("#breadcrumbLnk0")
        if breadcrumb:
            item["libraryCatalog"] = breadcrumb.get_text(strip=True)

        # Extract title from meta tags
        title_tag = doc.find("meta", {"property": "og:title"})
        if title_tag and title_tag.get("content"):
            title = title_tag["content"]
            # Handle "Replication Data for:" pattern
            if re.match(r"^(Replication )?[Dd]ata for:", title):
                match = re.match(r"(^(Replication )?[Dd]ata for:.*?)(:|$)", title)
                if match:
                    item["shortTitle"] = match.group(1)
            item["title"] = title

        # Extract DOI
        doi_tag = doc.find("meta", {"property": "citation_doi"})
        if doi_tag and doi_tag.get("content"):
            item["DOI"] = doi_tag["content"]

        # Extract repository
        repo_tag = doc.find("meta", {"property": "citation_publisher"})
        if repo_tag and repo_tag.get("content"):
            item["repository"] = repo_tag["content"]

        # Extract creators
        author_tags = doc.find_all("meta", {"property": "citation_author"})
        for author_tag in author_tags:
            author_name = author_tag.get("content", "")
            if author_name:
                names = author_name.split()
                if len(names) >= 2:
                    item["creators"].append(
                        {
                            "firstName": " ".join(names[:-1]),
                            "lastName": names[-1],
                            "creatorType": "author",
                        }
                    )
                else:
                    item["creators"].append(
                        {
                            "lastName": author_name,
                            "creatorType": "author",
                            "fieldMode": True,
                        }
                    )

        # Extract date
        date_tag = doc.find("meta", {"property": "citation_publication_date"})
        if date_tag and date_tag.get("content"):
            item["date"] = date_tag["content"]

        # Extract language
        lang_tag = doc.find("meta", {"property": "og:locale"})
        if lang_tag and lang_tag.get("content"):
            item["language"] = lang_tag["content"]

        # Extract keywords as tags
        keywords_tags = doc.find_all("meta", {"property": "citation_keywords"})
        for kw_tag in keywords_tags:
            keyword = kw_tag.get("content", "").strip()
            if keyword:
                item["tags"].append({"tag": keyword})

        # Add snapshot attachment
        item["attachments"].append(
            {"title": "Snapshot", "mimeType": "text/html", "url": url}
        )

        return item
