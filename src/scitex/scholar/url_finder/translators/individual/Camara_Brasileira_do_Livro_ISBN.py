"""
Câmara Brasileira do Livro ISBN Translator

Translates Brazilian books from ISBN search to Zotero format.

Metadata:
    translatorID: cdb5c893-ab69-4e96-9b5c-f4456d49ddd8
    label: Câmara Brasileira do Livro ISBN
    creator: Abe Jellinek
    target:
    minVersion: 5.0
    priority: 98
    inRepository: True
    translatorType: 8
    lastUpdated: 2023-09-26 16:11:18
"""

from typing import Dict, Any, List, Optional
import re
import json


class CamaraBrasileiraDoLivroISBNTranslator:
    """Translator for Brazilian ISBN database."""

    METADATA = {
        "translatorID": "cdb5c893-ab69-4e96-9b5c-f4456d49ddd8",
        "label": "Câmara Brasileira do Livro ISBN",
        "creator": "Abe Jellinek",
        "target": "",
        "minVersion": "5.0",
        "priority": 98,
        "inRepository": True,
        "translatorType": 8,
        "lastUpdated": "2023-09-26 16:11:18",
    }

    def detect_search(self, items: List[Dict[str, Any]]) -> bool:
        """
        Detect if items can be searched.

        Args:
            items: List of items with ISBN

        Returns:
            True if valid items found
        """
        cleaned_items = self._clean_data(items)
        return len(cleaned_items) > 0

    async def do_search(
        self, items: List[Dict[str, Any]], request_json_func
    ) -> List[Dict[str, Any]]:
        """
        Search for books by ISBN.

        Args:
            items: List of items with ISBN
            request_json_func: Function to make JSON requests

        Returns:
            List of Zotero items
        """
        items = self._clean_data(items)
        results = []

        for item in items:
            isbn = item.get("ISBN")
            if not isbn:
                continue

            search = isbn
            if len(isbn) == 10:
                search += " OR " + self._to_isbn13(isbn)

            body = {
                "count": True,
                "facets": [],
                "filter": "",
                "orderby": None,
                "queryType": "full",
                "search": search,
                "searchFields": "FormattedKey,RowKey",
                "searchMode": "any",
                "select": "*",
                "skip": 0,
                "top": 1,
            }

            headers = {
                "Content-Type": "application/json; charset=UTF-8",
                "api-key": "100216A23C5AEE390338BBD19EA86D29",
                "Origin": "https://www.cblservicos.org.br",
                "Referer": "https://www.cblservicos.org.br/",
            }

            response = await request_json_func(
                "https://isbn-search-br.search.windows.net/indexes/isbn-index/docs/search?api-version=2016-09-01",
                method="POST",
                headers=headers,
                body=json.dumps(body),
            )

            for result in response.get("value", []):
                results.append(self._translate_result(result))

        return results

    def _translate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate search result to Zotero item.

        Args:
            result: API result

        Returns:
            Zotero item dictionary
        """
        item = {"itemType": "book", "creators": [], "tags": [], "attachments": []}

        # Title
        title = result.get("Title", "")
        subtitle = result.get("Subtitle", "")
        if subtitle and ":" not in title and ":" not in subtitle:
            title += ": " + subtitle
        item["title"] = self._fix_case(title)

        # Other fields
        item["abstractNote"] = result.get("Sinopse", "")
        item["series"] = self._fix_case(result.get("Colection", ""))
        item["edition"] = result.get("Edicao", "")
        if item["edition"] == "1":
            item["edition"] = ""

        # Place
        cidade = result.get("Cidade", "")
        uf = result.get("UF", "")
        if cidade or uf:
            item["place"] = cidade + (", " + uf if uf else "")

        item["publisher"] = self._fix_case(result.get("Imprint", ""))
        item["date"] = self._str_to_iso(result.get("Date", ""))
        item["numPages"] = result.get("Paginas", "")
        if item["numPages"] == "0":
            item["numPages"] = ""

        # Language
        language = (
            result.get("IdiomasObra", [""])[0] if result.get("IdiomasObra") else "pt-BR"
        )
        if language == "português (Brasil)":
            language = "pt-BR"
        item["language"] = language

        item["ISBN"] = self._clean_isbn(result.get("FormattedKey", ""))

        # Authors
        authors = result.get("Authors", [])
        profissoes = result.get("Profissoes", [])

        for i, author in enumerate(authors):
            if author == author.upper():
                author = self._capitalize_name(author)

            creator_type = "author"
            if profissoes and len(profissoes) == len(authors):
                profissao = profissoes[i]
                if profissao in ["Coordenador", "Autor", "Roteirista"]:
                    creator_type = "author"
                elif profissao in ["Revisor", "Organizador", "Editor"]:
                    creator_type = "editor"
                elif profissao == "Tradutor":
                    creator_type = "translator"
                elif profissao in ["Ilustrador", "Projeto Gráfico"]:
                    creator_type = "illustrator"
                else:
                    creator_type = "author" if i == 0 else "contributor"
            elif i > 0:
                creator_type = "contributor"

            creator = self._clean_author(author, creator_type)
            item["creators"].append(creator)

        # Tags
        if result.get("Subject"):
            item["tags"].append({"tag": result["Subject"]})
        for tag in result.get("PalavrasChave", []):
            item["tags"].append({"tag": tag})

        return item

    def _fix_case(self, s: str) -> str:
        """Fix case of all-uppercase strings."""
        if s and s == s.upper():
            return self._capitalize_title(s)
        return s

    def _capitalize_title(self, s: str) -> str:
        """Capitalize title properly."""
        return " ".join(word.capitalize() for word in s.split())

    def _capitalize_name(self, s: str) -> str:
        """Capitalize name properly."""
        return " ".join(word.capitalize() for word in s.split())

    def _str_to_iso(self, date_str: str) -> str:
        """Convert date string to ISO format."""
        # Simple implementation for Brazilian dates
        return date_str

    def _clean_isbn(self, isbn: str) -> str:
        """Clean ISBN string."""
        return re.sub(r"[^0-9X]", "", isbn.upper())

    def _to_isbn13(self, isbn10: str) -> str:
        """Convert ISBN-10 to ISBN-13."""
        if len(isbn10) != 10:
            return isbn10
        isbn13 = "978" + isbn10[:-1]
        check_sum = sum((3 if i % 2 else 1) * int(c) for i, c in enumerate(isbn13))
        check_digit = (10 - (check_sum % 10)) % 10
        return isbn13 + str(check_digit)

    def _clean_author(self, name: str, creator_type: str) -> Dict[str, Any]:
        """Parse author name."""
        name = name.strip()
        has_comma = "," in name

        if has_comma:
            parts = name.split(",", 1)
            last_name = parts[0].strip()
            first_name = parts[1].strip() if len(parts) > 1 else ""

            # Handle suffixes
            suffixes = ["filho", "junior", "neto", "sobrinho", "segundo", "terceiro"]
            last_name_lower = self._remove_diacritics(last_name.lower())
            if last_name_lower in suffixes and first_name:
                first_parts = first_name.split()
                if first_parts:
                    last_name = first_parts[-1] + " " + last_name
                    first_name = " ".join(first_parts[:-1])

            if first_name:
                return {
                    "firstName": first_name,
                    "lastName": last_name,
                    "creatorType": creator_type,
                }
            else:
                return {
                    "lastName": last_name,
                    "creatorType": creator_type,
                    "fieldMode": True,
                }
        else:
            return {"lastName": name, "creatorType": creator_type, "fieldMode": True}

    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritics from text."""
        import unicodedata

        return "".join(
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def _clean_data(self, items: Any) -> List[Dict[str, Any]]:
        """Clean and filter items."""
        if not isinstance(items, list):
            items = [items]

        result = []
        for item in items:
            if isinstance(item, str):
                item = {"ISBN": item}

            if item.get("ISBN"):
                isbn = self._clean_isbn(item["ISBN"])
                # Filter for Brazilian ISBNs
                if (
                    isbn.startswith("97865")
                    or isbn.startswith("65")
                    or isbn.startswith("97885")
                    or isbn.startswith("85")
                ):
                    item["ISBN"] = isbn
                    result.append(item)

        return result
