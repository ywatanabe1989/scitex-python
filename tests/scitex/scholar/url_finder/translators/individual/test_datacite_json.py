# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/datacite_json.py
# --------------------------------------------------------------------------------
# """
# Datacite JSON Translator
# 
# Imports DataCite JSON format metadata.
# 
# Metadata:
#     translatorID: b5b5808b-1c61-473d-9a02-e1f5ba7b8eef
#     label: Datacite JSON
#     creator: Philipp Zumstein
#     target: json
#     minVersion: 3.0
#     priority: 100
#     inRepository: True
#     translatorType: 1
#     lastUpdated: 2025-04-29 03:02:00
# """
# 
# from typing import Dict, Any, Optional
# import json
# 
# 
# class DataciteJSONTranslator:
#     """Import translator for DataCite JSON format."""
# 
#     METADATA = {
#         "translatorID": "b5b5808b-1c61-473d-9a02-e1f5ba7b8eef",
#         "label": "Datacite JSON",
#         "creator": "Philipp Zumstein",
#         "target": "json",
#         "minVersion": "3.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 1,  # Import type
#         "lastUpdated": "2025-04-29 03:02:00",
#     }
# 
#     # Type mappings from DataCite/CSL to Zotero
#     TYPE_MAPPINGS = {
#         "article": "preprint",
#         "book": "book",
#         "chapter": "bookSection",
#         "article-journal": "journalArticle",
#         "article-magazine": "magazineArticle",
#         "article-newspaper": "newspaperArticle",
#         "thesis": "thesis",
#         "entry-encyclopedia": "encyclopediaArticle",
#         "entry-dictionary": "dictionaryEntry",
#         "paper-conference": "conferencePaper",
#         "personal_communication": "letter",
#         "manuscript": "manuscript",
#         "interview": "interview",
#         "motion_picture": "film",
#         "graphic": "artwork",
#         "webpage": "webpage",
#         "report": "report",
#         "bill": "bill",
#         "legal_case": "case",
#         "patent": "patent",
#         "legislation": "statute",
#         "map": "map",
#         "post-weblog": "blogPost",
#         "post": "forumPost",
#         "song": "audioRecording",
#         "speech": "presentation",
#         "broadcast": "radioBroadcast",
#         "dataset": "dataset",
#     }
# 
#     def detect_import(self, content: str) -> bool:
#         """
#         Detect if the input is DataCite JSON.
# 
#         Args:
#             content: JSON string to check
# 
#         Returns:
#             True if this is DataCite JSON
#         """
#         try:
#             data = json.loads(content)
#             # Check for DataCite schema version or agency
#             if data.get("schemaVersion", "").startswith("http://datacite.org/schema/"):
#                 return True
#             if "datacite" in data.get("agency", "").lower():
#                 return True
#             return False
#         except (json.JSONDecodeError, AttributeError):
#             return False
# 
#     def do_import(self, content: str) -> Dict[str, Any]:
#         """
#         Import DataCite JSON and convert to Zotero item.
# 
#         Args:
#             content: JSON string containing DataCite metadata
# 
#         Returns:
#             Dictionary containing Zotero item data
#         """
#         try:
#             data = json.loads(content)
#         except json.JSONDecodeError:
#             return {}
# 
#         # Determine item type
#         item_type = "journalArticle"  # default
# 
#         if data.get("types"):
#             types_info = data["types"]
# 
#             # Check citeproc type
#             if (
#                 types_info.get("citeproc")
#                 and types_info["citeproc"] in self.TYPE_MAPPINGS
#             ):
#                 item_type = self.TYPE_MAPPINGS[types_info["citeproc"]]
# 
#             # Check for software types
#             schema_org = types_info.get("schemaOrg", "").lower()
#             if schema_org in [
#                 "softwaresourcecode",
#                 "softwareapplication",
#                 "mobileapplication",
#                 "videogame",
#                 "webapplication",
#             ]:
#                 item_type = "computerProgram"
# 
#             # Special case for book chapters
#             if types_info.get("resourceTypeGeneral") == "BookChapter":
#                 item_type = "bookSection"
# 
#         item = {"itemType": item_type, "creators": [], "tags": [], "attachments": []}
# 
#         # Handle dataset type for older Zotero versions
#         if (
#             data.get("types", {}).get("citeproc") == "dataset"
#             and item_type == "document"
#         ):
#             item["extra"] = "Type: dataset"
# 
#         # Extract titles
#         title = ""
#         alternate_title = ""
#         for title_element in data.get("titles", []):
#             if not title_element.get("title"):
#                 continue
# 
#             title_type = title_element.get("titleType", "").lower()
#             if not title_type:
#                 title = title_element["title"] + title
#             elif title_type == "subtitle":
#                 title = title + ": " + title_element["title"]
#             elif not alternate_title:
#                 alternate_title = title_element["title"]
# 
#         item["title"] = title or alternate_title
# 
#         # Extract creators
#         for creator in data.get("creators", []):
#             if creator.get("familyName") and creator.get("givenName"):
#                 item["creators"].append(
#                     {
#                         "lastName": creator["familyName"],
#                         "firstName": creator["givenName"],
#                         "creatorType": "author",
#                     }
#                 )
#             elif creator.get("nameType") == "Personal":
#                 name = creator.get("name", "")
#                 names = name.split()
#                 if len(names) >= 2:
#                     item["creators"].append(
#                         {
#                             "lastName": names[-1],
#                             "firstName": " ".join(names[:-1]),
#                             "creatorType": "author",
#                         }
#                     )
#                 else:
#                     item["creators"].append(
#                         {"lastName": name, "creatorType": "author", "fieldMode": True}
#                     )
#             else:
#                 item["creators"].append(
#                     {
#                         "lastName": creator.get("name", ""),
#                         "creatorType": "author",
#                         "fieldMode": True,
#                     }
#                 )
# 
#         # Extract contributors
#         for contributor in data.get("contributors", []):
#             role = "contributor"
#             contrib_type = contributor.get("contributorType", "").lower()
#             if contrib_type == "editor":
#                 role = "editor"
#             elif contrib_type == "producer":
#                 role = "producer"
# 
#             if contributor.get("familyName") and contributor.get("givenName"):
#                 item["creators"].append(
#                     {
#                         "lastName": contributor["familyName"],
#                         "firstName": contributor["givenName"],
#                         "creatorType": role,
#                     }
#                 )
#             elif contributor.get("nameType") == "Personal":
#                 name = contributor.get("name", "")
#                 names = name.split()
#                 if len(names) >= 2:
#                     item["creators"].append(
#                         {
#                             "lastName": names[-1],
#                             "firstName": " ".join(names[:-1]),
#                             "creatorType": role,
#                         }
#                     )
#                 else:
#                     item["creators"].append(
#                         {"lastName": name, "creatorType": role, "fieldMode": True}
#                     )
#             else:
#                 item["creators"].append(
#                     {
#                         "lastName": contributor.get("name", ""),
#                         "creatorType": role,
#                         "fieldMode": True,
#                     }
#                 )
# 
#         # Extract publisher
#         publisher = data.get("publisher")
#         if isinstance(publisher, dict):
#             item["publisher"] = publisher.get("name", "")
#         else:
#             item["publisher"] = publisher
# 
#         # Extract DOI
#         if data.get("doi"):
#             item["DOI"] = data["doi"]
# 
#         # Extract URL
#         if data.get("url"):
#             item["url"] = data["url"]
# 
#         # Extract dates
#         for date_info in data.get("dates", []):
#             date_type = date_info.get("dateType", "").lower()
#             date_value = date_info.get("date", "")
# 
#             if date_type == "issued" and date_value:
#                 item["date"] = date_value[:10] if len(date_value) >= 10 else date_value
# 
#         # Extract publication year
#         if data.get("publicationYear"):
#             if not item.get("date"):
#                 item["date"] = str(data["publicationYear"])
# 
#         # Extract descriptions (abstract)
#         for desc in data.get("descriptions", []):
#             if desc.get("descriptionType", "").lower() == "abstract":
#                 item["abstractNote"] = desc.get("description", "")
#                 break
# 
#         # Extract subjects as tags
#         for subject in data.get("subjects", []):
#             subject_text = subject.get("subject", "").strip()
#             if subject_text:
#                 item["tags"].append({"tag": subject_text})
# 
#         # Extract language
#         if data.get("language"):
#             item["language"] = data["language"]
# 
#         return item

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/datacite_json.py
# --------------------------------------------------------------------------------
