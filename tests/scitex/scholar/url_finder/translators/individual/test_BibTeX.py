# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/BibTeX.py
# --------------------------------------------------------------------------------
# """
# BibTeX Translator
# 
# Imports and exports BibTeX format. This is a simplified Python implementation
# of the complex JavaScript BibTeX translator.
# 
# Note: This is a bidirectional translator (type 3 = import+export).
# 
# Metadata:
#     translatorID: 9cb70025-a888-4a29-a210-93ec52da40d4
#     label: BibTeX
#     creator: Simon Kornblith, Richard Karnesky, and Emiliano Heyns
#     target: bib
#     minVersion: 3.0
#     priority: 100
#     inRepository: True
#     translatorType: 3
#     lastUpdated: 2024-05-23 02:08:36
# """
# 
# from typing import Dict, Any, List, Optional
# import re
# 
# 
# class BibTeXTranslator:
#     """Translator for BibTeX format (simplified implementation)."""
# 
#     METADATA = {
#         "translatorID": "9cb70025-a888-4a29-a210-93ec52da40d4",
#         "label": "BibTeX",
#         "creator": "Simon Kornblith, Richard Karnesky, and Emiliano Heyns",
#         "target": "bib",
#         "minVersion": "3.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 3,  # Import + Export
#         "lastUpdated": "2024-05-23 02:08:36",
#     }
# 
#     # Item type mappings (simplified)
#     TYPE_MAP = {
#         "book": "book",
#         "bookSection": "incollection",
#         "journalArticle": "article",
#         "magazineArticle": "article",
#         "newspaperArticle": "article",
#         "thesis": "phdthesis",
#         "manuscript": "unpublished",
#         "conferencePaper": "inproceedings",
#         "report": "techreport",
#         "patent": "misc",
#         "webpage": "misc",
#         "document": "misc",
#     }
# 
#     def __init__(self):
#         """Initialize the translator."""
#         self.cite_keys = set()
# 
#     def detect_import(self, text: str) -> bool:
#         """
#         Detect if the text is BibTeX format.
# 
#         Args:
#             text: Text to check
# 
#         Returns:
#             True if BibTeX format is detected
#         """
#         # Look for BibTeX entry pattern: @type{key,
#         pattern = r"^\s*@[a-zA-Z]+[\(\{]"
#         return bool(re.search(pattern, text, re.MULTILINE))
# 
#     def do_import(self, text: str) -> List[Dict[str, Any]]:
#         """
#         Import BibTeX text to Zotero items.
# 
#         Args:
#             text: BibTeX formatted text
# 
#         Returns:
#             List of Zotero items
#         """
#         items = []
# 
#         # Pattern to match BibTeX entries
#         entry_pattern = r"@(\w+)\s*\{([^,]+),\s*([^}]+)\}"
# 
#         for match in re.finditer(entry_pattern, text, re.DOTALL):
#             entry_type, cite_key, fields_text = match.groups()
# 
#             item = self._parse_entry(entry_type, cite_key, fields_text)
#             if item:
#                 items.append(item)
# 
#         return items
# 
#     def do_export(self, items: List[Dict[str, Any]]) -> str:
#         """
#         Export items to BibTeX format.
# 
#         Args:
#             items: List of Zotero items to export
# 
#         Returns:
#             BibTeX formatted string
#         """
#         output = []
# 
#         for item in items:
#             if item.get("itemType") in ["note", "attachment"]:
#                 continue
# 
#             entry = self._export_item(item)
#             if entry:
#                 output.append(entry)
# 
#         return "\n\n".join(output)
# 
#     def _parse_entry(
#         self, entry_type: str, cite_key: str, fields_text: str
#     ) -> Optional[Dict[str, Any]]:
#         """
#         Parse a single BibTeX entry.
# 
#         Args:
#             entry_type: BibTeX entry type
#             cite_key: Citation key
#             fields_text: Fields as text
# 
#         Returns:
#             Zotero item dictionary
#         """
#         # Map BibTeX type to Zotero type (reverse lookup)
#         zotero_type = self._bibtex_to_zotero_type(entry_type.lower())
# 
#         item = {
#             "itemType": zotero_type,
#             "citationKey": cite_key,
#             "creators": [],
#             "tags": [],
#             "attachments": [],
#         }
# 
#         # Parse fields
#         field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}|(\w+)\s*=\s*"([^"]*)"'
# 
#         for match in re.finditer(field_pattern, fields_text):
#             if match.group(1):
#                 field_name = match.group(1).lower()
#                 field_value = match.group(2)
#             else:
#                 field_name = match.group(3).lower()
#                 field_value = match.group(4)
# 
#             self._process_field(item, field_name, field_value)
# 
#         return item
# 
#     def _bibtex_to_zotero_type(self, bibtex_type: str) -> str:
#         """Map BibTeX type to Zotero type."""
#         type_map = {
#             "article": "journalArticle",
#             "book": "book",
#             "inproceedings": "conferencePaper",
#             "incollection": "bookSection",
#             "phdthesis": "thesis",
#             "mastersthesis": "thesis",
#             "techreport": "report",
#             "unpublished": "manuscript",
#             "misc": "document",
#         }
#         return type_map.get(bibtex_type, "document")
# 
#     def _process_field(self, item: Dict[str, Any], field_name: str, field_value: str):
#         """Process a BibTeX field and add to item."""
#         field_map = {
#             "title": "title",
#             "journal": "publicationTitle",
#             "booktitle": "publicationTitle",
#             "year": "date",
#             "volume": "volume",
#             "number": "issue",
#             "pages": "pages",
#             "doi": "DOI",
#             "isbn": "ISBN",
#             "issn": "ISSN",
#             "url": "url",
#             "abstract": "abstractNote",
#             "publisher": "publisher",
#             "address": "place",
#             "edition": "edition",
#             "series": "series",
#         }
# 
#         if field_name == "author" or field_name == "editor":
#             # Parse authors/editors
#             creators = self._parse_creators(field_value, field_name)
#             item["creators"].extend(creators)
#         elif field_name == "keywords":
#             # Parse keywords as tags
#             keywords = field_value.split(",")
#             item["tags"].extend({"tag": k.strip()} for k in keywords if k.strip())
#         elif field_name in field_map:
#             item[field_map[field_name]] = field_value.strip()
# 
#     def _parse_creators(
#         self, creators_text: str, creator_type: str
#     ) -> List[Dict[str, Any]]:
#         """Parse BibTeX author/editor field."""
#         creators = []
# 
#         # Split by " and "
#         names = creators_text.split(" and ")
# 
#         for name in names:
#             name = name.strip()
#             if not name:
#                 continue
# 
#             # Handle "Last, First" format
#             if "," in name:
#                 parts = name.split(",", 1)
#                 creator = {
#                     "lastName": parts[0].strip(),
#                     "firstName": parts[1].strip(),
#                     "creatorType": creator_type,
#                 }
#             else:
#                 # Handle "First Last" format
#                 parts = name.split()
#                 if len(parts) >= 2:
#                     creator = {
#                         "firstName": " ".join(parts[:-1]),
#                         "lastName": parts[-1],
#                         "creatorType": creator_type,
#                     }
#                 else:
#                     creator = {
#                         "lastName": name,
#                         "creatorType": creator_type,
#                         "fieldMode": True,
#                     }
# 
#             creators.append(creator)
# 
#         return creators
# 
#     def _export_item(self, item: Dict[str, Any]) -> str:
#         """Export a single item to BibTeX format."""
#         item_type = item.get("itemType", "misc")
#         bib_type = self.TYPE_MAP.get(item_type, "misc")
# 
#         cite_key = self._build_cite_key(item)
# 
#         lines = [f"@{bib_type}{{{cite_key}"]
# 
#         # Add fields
#         if "title" in item:
#             lines.append(f",\n\ttitle = {{{item['title']}}}")
# 
#         if "creators" in item:
#             authors, editors = self._format_creators(item["creators"])
#             if authors:
#                 lines.append(f",\n\tauthor = {{{authors}}}")
#             if editors:
#                 lines.append(f",\n\teditor = {{{editors}}}")
# 
#         if "publicationTitle" in item:
#             field = "journal" if item_type == "journalArticle" else "booktitle"
#             lines.append(f",\n\t{field} = {{{item['publicationTitle']}}}")
# 
#         if "date" in item:
#             year_match = re.search(r"\d{4}", item["date"])
#             if year_match:
#                 lines.append(f",\n\tyear = {{{year_match.group(0)}}}")
# 
#         for field in ["volume", "pages", "DOI", "ISBN", "ISSN", "url", "publisher"]:
#             if field in item and item[field]:
#                 bib_field = field.lower()
#                 lines.append(f",\n\t{bib_field} = {{{item[field]}}}")
# 
#         lines.append("\n}")
#         return "".join(lines)
# 
#     def _build_cite_key(self, item: Dict[str, Any]) -> str:
#         """Build citation key."""
#         parts = []
# 
#         if "creators" in item and item["creators"]:
#             last_name = item["creators"][0].get("lastName", "noauthor")
#             parts.append(last_name.lower().replace(" ", ""))
#         else:
#             parts.append("noauthor")
# 
#         if "date" in item:
#             year_match = re.search(r"\d{4}", item["date"])
#             if year_match:
#                 parts.append(year_match.group(0))
# 
#         base_key = "_".join(parts) if parts else "unknown"
# 
#         cite_key = base_key
#         counter = 1
#         while cite_key in self.cite_keys:
#             cite_key = f"{base_key}_{counter}"
#             counter += 1
# 
#         self.cite_keys.add(cite_key)
#         return cite_key
# 
#     def _format_creators(self, creators: List[Dict[str, Any]]) -> tuple:
#         """Format creators, returning (authors, editors)."""
#         authors = []
#         editors = []
# 
#         for creator in creators:
#             if creator.get("firstName") and creator.get("lastName"):
#                 name = f"{creator['lastName']}, {creator['firstName']}"
#             elif creator.get("lastName"):
#                 name = creator["lastName"]
#             else:
#                 continue
# 
#             if creator.get("creatorType") == "editor":
#                 editors.append(name)
#             else:
#                 authors.append(name)
# 
#         return (" and ".join(authors), " and ".join(editors))

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/BibTeX.py
# --------------------------------------------------------------------------------
