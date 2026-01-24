#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/storage/_mixins/_metadata_conversion.py

"""
Metadata conversion mixin for LibraryManager.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Dict

from scitex.scholar.metadata_engines.utils import BASE_STRUCTURE


class MetadataConversionMixin:
    """Mixin providing metadata conversion methods."""

    def _dotdict_to_dict(self, obj):
        """Recursively convert DotDict to plain dict for JSON serialization."""
        from scitex.dict import DotDict

        if isinstance(obj, DotDict):
            return {k: self._dotdict_to_dict(v) for k, v in obj._data.items()}
        elif isinstance(obj, dict):
            return {k: self._dotdict_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._dotdict_to_dict(item) for item in obj]
        else:
            return obj

    def _add_engine_to_list(self, engines_list: list, source: str) -> None:
        """Helper to add source to engines list if not already present."""
        if source and source not in engines_list:
            engines_list.append(source)

    def _convert_to_standardized_metadata(self, flat_metadata: Dict) -> OrderedDict:
        """Convert flat metadata dict to standardized nested structure."""
        standardized = copy.deepcopy(BASE_STRUCTURE)

        # ID section
        if "doi" in flat_metadata:
            standardized["id"]["doi"] = flat_metadata["doi"]
            self._add_engine_to_list(
                standardized["id"]["doi_engines"],
                flat_metadata.get("doi_source"),
            )
        if "scitex_id" in flat_metadata:
            standardized["id"]["scholar_id"] = flat_metadata["scitex_id"]

        # Basic section
        if "title" in flat_metadata:
            standardized["basic"]["title"] = flat_metadata["title"]
            self._add_engine_to_list(
                standardized["basic"]["title_engines"],
                flat_metadata.get("title_source"),
            )
        if "authors" in flat_metadata:
            standardized["basic"]["authors"] = flat_metadata["authors"]
            self._add_engine_to_list(
                standardized["basic"]["authors_engines"],
                flat_metadata.get("authors_source"),
            )
        if "year" in flat_metadata:
            standardized["basic"]["year"] = flat_metadata["year"]
            self._add_engine_to_list(
                standardized["basic"]["year_engines"],
                flat_metadata.get("year_source"),
            )
        if "abstract" in flat_metadata:
            standardized["basic"]["abstract"] = flat_metadata["abstract"]
            self._add_engine_to_list(
                standardized["basic"]["abstract_engines"],
                flat_metadata.get("abstract_source"),
            )

        # Citation count section
        self._convert_citation_count(flat_metadata, standardized)

        # Publication section
        self._convert_publication_fields(flat_metadata, standardized)

        # URL section
        self._convert_url_fields(flat_metadata, standardized)

        # Path section
        if "pdf_path" in flat_metadata:
            standardized["path"]["pdfs"] = [flat_metadata["pdf_path"]]
            self._add_engine_to_list(
                standardized["path"]["pdfs_engines"],
                "ScholarPDFDownloaderWithScreenshotsParallel",
            )

        return standardized

    def _convert_citation_count(
        self, flat_metadata: Dict, standardized: OrderedDict
    ) -> None:
        """Convert citation count fields to standardized format."""
        if "citation_count" not in flat_metadata:
            return

        cc_value = flat_metadata["citation_count"]
        if isinstance(cc_value, dict):
            standardized["citation_count"]["total"] = cc_value.get("total")
            self._add_engine_to_list(
                standardized["citation_count"]["total_engines"],
                cc_value.get("total_source"),
            )
            for year in [
                "2025",
                "2024",
                "2023",
                "2022",
                "2021",
                "2020",
                "2019",
                "2018",
                "2017",
                "2016",
                "2015",
            ]:
                if year in cc_value:
                    standardized["citation_count"][year] = cc_value[year]
                    if f"{year}_source" in cc_value:
                        self._add_engine_to_list(
                            standardized["citation_count"][f"{year}_engines"],
                            cc_value.get(f"{year}_source"),
                        )
        else:
            standardized["citation_count"]["total"] = cc_value
            self._add_engine_to_list(
                standardized["citation_count"]["total_engines"],
                flat_metadata.get("citation_count_source"),
            )

    def _convert_publication_fields(
        self, flat_metadata: Dict, standardized: OrderedDict
    ) -> None:
        """Convert publication fields to standardized format."""
        if "journal" in flat_metadata:
            standardized["publication"]["journal"] = flat_metadata["journal"]
            self._add_engine_to_list(
                standardized["publication"]["journal_engines"],
                flat_metadata.get("journal_source"),
            )
        if "short_journal" in flat_metadata:
            standardized["publication"]["short_journal"] = flat_metadata[
                "short_journal"
            ]
        if "impact_factor" in flat_metadata:
            standardized["publication"]["impact_factor"] = flat_metadata[
                "impact_factor"
            ]
        if "issn" in flat_metadata:
            standardized["publication"]["issn"] = flat_metadata["issn"]
        if "volume" in flat_metadata:
            standardized["publication"]["volume"] = flat_metadata["volume"]
        if "issue" in flat_metadata:
            standardized["publication"]["issue"] = flat_metadata["issue"]
        if "pages" in flat_metadata:
            pages = flat_metadata["pages"]
            if pages and "-" in str(pages):
                first, last = str(pages).split("-", 1)
                standardized["publication"]["first_page"] = first.strip()
                standardized["publication"]["last_page"] = last.strip()
        if "publisher" in flat_metadata:
            standardized["publication"]["publisher"] = flat_metadata["publisher"]

    def _convert_url_fields(
        self, flat_metadata: Dict, standardized: OrderedDict
    ) -> None:
        """Convert URL fields to standardized format."""
        if "url_doi" in flat_metadata:
            standardized["url"]["doi"] = flat_metadata["url_doi"]
        if "url_publisher" in flat_metadata:
            standardized["url"]["publisher"] = flat_metadata["url_publisher"]
            self._add_engine_to_list(
                standardized["url"]["publisher_engines"], "ScholarURLFinder"
            )
        if "url_openurl_query" in flat_metadata:
            standardized["url"]["openurl_query"] = flat_metadata["url_openurl_query"]
        if "url_openurl_resolved" in flat_metadata:
            standardized["url"]["openurl_resolved"] = flat_metadata[
                "url_openurl_resolved"
            ]
            self._add_engine_to_list(
                standardized["url"]["openurl_resolved_engines"], "ScholarURLFinder"
            )
        if "urls_pdf" in flat_metadata:
            standardized["url"]["pdfs"] = flat_metadata["urls_pdf"]
            self._add_engine_to_list(
                standardized["url"]["pdfs_engines"], "ScholarURLFinder"
            )

    def _call_path_manager_get_storage_paths(
        self, paper_info: Dict, collection_name: str = "MASTER"
    ) -> Dict[str, Any]:
        """Helper to call PathManager's get_paper_storage_paths with proper parameters."""
        doi = paper_info.get("doi")
        title = paper_info.get("title")
        authors = paper_info.get("authors", [])
        year = paper_info.get("year")
        journal = paper_info.get("journal")

        storage_path, readable_name, paper_id = (
            self.config.path_manager.get_paper_storage_paths(
                doi=doi,
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                project=collection_name,
            )
        )

        return {
            "storage_path": storage_path,
            "readable_name": readable_name,
            "unique_id": paper_id,
        }


# EOF
