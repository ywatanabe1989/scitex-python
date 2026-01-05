# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/BibLaTeX.py
# --------------------------------------------------------------------------------
# """
# BibLaTeX Translator
# 
# Exports Zotero items to BibLaTeX format.
# 
# Note: This is an export translator (type 2), not a web scraper.
# It converts Zotero items to BibLaTeX bibliography format.
# 
# Metadata:
#     translatorID: b6e39b57-8942-4d11-8259-342c46ce395f
#     label: BibLaTeX
#     creator: Simon Kornblith, Richard Karnesky and Anders Johansson
#     target: bib
#     minVersion: 2.1.9
#     priority: 100
#     inRepository: True
#     translatorType: 2
#     lastUpdated: 2024-03-25 14:49:42
# """
# 
# from typing import Dict, Any, List, Optional
# import re
# 
# 
# class BibLaTeXTranslator:
#     """Translator for BibLaTeX export format."""
# 
#     METADATA = {
#         "translatorID": "b6e39b57-8942-4d11-8259-342c46ce395f",
#         "label": "BibLaTeX",
#         "creator": "Simon Kornblith, Richard Karnesky and Anders Johansson",
#         "target": "bib",
#         "minVersion": "2.1.9",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 2,  # Export translator
#         "lastUpdated": "2024-03-25 14:49:42",
#     }
# 
#     # Field mappings from Zotero to BibLaTeX
#     FIELD_MAP = {
#         "place": "location",
#         "chapter": "chapter",
#         "edition": "edition",
#         "title": "title",
#         "volume": "volume",
#         "rights": "rights",
#         "ISBN": "isbn",
#         "ISSN": "issn",
#         "url": "url",
#         "DOI": "doi",
#         "series": "series",
#         "shortTitle": "shorttitle",
#         "assignee": "holder",
#         "abstractNote": "abstract",
#         "numberOfVolumes": "volumes",
#         "version": "version",
#         "conferenceName": "eventtitle",
#         "pages": "pages",
#         "numPages": "pagetotal",
#     }
# 
#     # Item type mappings from Zotero to BibLaTeX
#     TYPE_MAP = {
#         "book": "book",
#         "bookSection": "incollection",
#         "journalArticle": "article",
#         "magazineArticle": "article",
#         "newspaperArticle": "article",
#         "thesis": "thesis",
#         "letter": "letter",
#         "manuscript": "unpublished",
#         "interview": "misc",
#         "film": "movie",
#         "artwork": "artwork",
#         "webpage": "online",
#         "conferencePaper": "inproceedings",
#         "report": "report",
#         "bill": "legislation",
#         "case": "jurisdiction",
#         "hearing": "jurisdiction",
#         "patent": "patent",
#         "statute": "legislation",
#         "email": "letter",
#         "map": "misc",
#         "blogPost": "online",
#         "instantMessage": "misc",
#         "forumPost": "online",
#         "audioRecording": "audio",
#         "presentation": "unpublished",
#         "videoRecording": "video",
#         "tvBroadcast": "misc",
#         "radioBroadcast": "misc",
#         "podcast": "audio",
#         "computerProgram": "software",
#         "document": "misc",
#         "encyclopediaArticle": "inreference",
#         "dictionaryEntry": "inreference",
#     }
# 
#     def __init__(self):
#         """Initialize the translator."""
#         self.cite_keys = set()
# 
#     def do_export(self, items: List[Dict[str, Any]]) -> str:
#         """
#         Export items to BibLaTeX format.
# 
#         Args:
#             items: List of Zotero items to export
# 
#         Returns:
#             BibLaTeX formatted string
#         """
#         output = []
# 
#         for item in items:
#             # Skip notes and attachments
#             if item.get("itemType") in ["note", "attachment"]:
#                 continue
# 
#             entry = self._export_item(item)
#             if entry:
#                 output.append(entry)
# 
#         return "\n\n".join(output)
# 
#     def _export_item(self, item: Dict[str, Any]) -> str:
#         """
#         Export a single item to BibLaTeX format.
# 
#         Args:
#             item: Zotero item
# 
#         Returns:
#             BibLaTeX entry string
#         """
#         # Determine entry type
#         item_type = item.get("itemType", "misc")
#         bib_type = self.TYPE_MAP.get(item_type, "misc")
# 
#         # Generate cite key
#         cite_key = self._build_cite_key(item)
# 
#         # Start entry
#         lines = [f"@{bib_type}{{{cite_key}"]
# 
#         # Add fields
#         for zotero_field, biblatex_field in self.FIELD_MAP.items():
#             if zotero_field in item and item[zotero_field]:
#                 value = self._escape_value(item[zotero_field])
#                 lines.append(f",\n\t{biblatex_field} = {{{value}}}")
# 
#         # Add special handling for publication title
#         if "publicationTitle" in item and item["publicationTitle"]:
#             if item_type in [
#                 "bookSection",
#                 "conferencePaper",
#                 "dictionaryEntry",
#                 "encyclopediaArticle",
#             ]:
#                 field_name = "booktitle"
#             elif item_type in ["magazineArticle", "newspaperArticle", "journalArticle"]:
#                 field_name = "journaltitle"
#             else:
#                 field_name = "journaltitle"
# 
#             value = self._escape_value(item["publicationTitle"])
#             lines.append(f",\n\t{field_name} = {{{value}}}")
# 
#         # Add creators
#         if "creators" in item and item["creators"]:
#             creators_str = self._format_creators(item["creators"])
#             if creators_str:
#                 lines.append(creators_str)
# 
#         # Add date
#         if "date" in item and item["date"]:
#             lines.append(f",\n\tdate = {{{item['date']}}}")
# 
#         # Add tags as keywords
#         if "tags" in item and item["tags"]:
#             keywords = ", ".join(
#                 tag.get("tag", "") for tag in item["tags"] if tag.get("tag")
#             )
#             if keywords:
#                 lines.append(f",\n\tkeywords = {{{keywords}}}")
# 
#         # Close entry
#         lines.append(",\n}")
# 
#         return "".join(lines)
# 
#     def _build_cite_key(self, item: Dict[str, Any]) -> str:
#         """
#         Build citation key for an item.
# 
#         Args:
#             item: Zotero item
# 
#         Returns:
#             Citation key string
#         """
#         # Simple cite key: author_title_year
#         parts = []
# 
#         # Add first author last name
#         if "creators" in item and item["creators"]:
#             first_creator = item["creators"][0]
#             last_name = first_creator.get("lastName", "noauthor")
#             parts.append(self._clean_cite_key_part(last_name))
#         else:
#             parts.append("noauthor")
# 
#         # Add first word of title
#         if "title" in item and item["title"]:
#             title_words = item["title"].split()
#             if title_words:
#                 parts.append(self._clean_cite_key_part(title_words[0]))
#         else:
#             parts.append("notitle")
# 
#         # Add year
#         if "date" in item and item["date"]:
#             year_match = re.search(r"\d{4}", item["date"])
#             if year_match:
#                 parts.append(year_match.group(0))
#             else:
#                 parts.append("nodate")
#         else:
#             parts.append("nodate")
# 
#         base_key = "_".join(parts)
# 
#         # Ensure uniqueness
#         cite_key = base_key
#         counter = 1
#         while cite_key in self.cite_keys:
#             cite_key = f"{base_key}_{counter}"
#             counter += 1
# 
#         self.cite_keys.add(cite_key)
#         return cite_key
# 
#     def _clean_cite_key_part(self, text: str) -> str:
#         """
#         Clean a part of the citation key.
# 
#         Args:
#             text: Text to clean
# 
#         Returns:
#             Cleaned text suitable for citation key
#         """
#         # Remove special characters, keep only alphanumeric and underscore
#         text = text.lower()
#         text = re.sub(r"[^a-z0-9_-]", "", text)
#         return text
# 
#     def _format_creators(self, creators: List[Dict[str, Any]]) -> str:
#         """
#         Format creators for BibLaTeX.
# 
#         Args:
#             creators: List of creator dictionaries
# 
#         Returns:
#             Formatted creator string
#         """
#         authors = []
#         editors = []
# 
#         for creator in creators:
#             creator_type = creator.get("creatorType", "author")
# 
#             # Format name
#             if creator.get("firstName") and creator.get("lastName"):
#                 name = f"{creator['lastName']}, {creator['firstName']}"
#             elif creator.get("lastName"):
#                 name = creator["lastName"]
#             else:
#                 continue
# 
#             if creator_type in ["author", "inventor", "artist", "programmer"]:
#                 authors.append(name)
#             elif creator_type == "editor":
#                 editors.append(name)
# 
#         result = []
#         if authors:
#             author_str = " and ".join(authors)
#             result.append(f",\n\tauthor = {{{author_str}}}")
#         if editors:
#             editor_str = " and ".join(editors)
#             result.append(f",\n\teditor = {{{editor_str}}}")
# 
#         return "".join(result)
# 
#     def _escape_value(self, value: Any) -> str:
#         """
#         Escape special characters in BibLaTeX values.
# 
#         Args:
#             value: Value to escape
# 
#         Returns:
#             Escaped value
#         """
#         if not isinstance(value, str):
#             value = str(value)
# 
#         # Escape special LaTeX characters
#         replacements = {
#             "\\": "\\textbackslash{}",
#             "{": "\\{",
#             "}": "\\}",
#             "#": "\\#",
#             "$": "\\$",
#             "%": "\\%",
#             "&": "\\&",
#             "_": "\\_",
#             "~": "\\textasciitilde{}",
#             "^": "\\textasciicircum{}",
#         }
# 
#         for char, replacement in replacements.items():
#             value = value.replace(char, replacement)
# 
#         return value

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/BibLaTeX.py
# --------------------------------------------------------------------------------
