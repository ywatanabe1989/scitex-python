#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-11 15:15:00
# File: /home/ywatanabe/proj/scitex-code/src/scitex/msword/reader.py

"""
DOCX -> SciTeX writer document converter.

This module reads MS Word .docx files and converts them into
SciTeX's intermediate document format for further processing.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .profiles import BaseWordProfile

# Lazy import for python-docx
try:
    import docx
    from docx.document import Document as DocxDocument
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    DOCX_AVAILABLE = True
    _DOCX_IMPORT_ERROR = None
except ImportError as exc:
    DOCX_AVAILABLE = False
    _DOCX_IMPORT_ERROR = exc
    DocxDocument = None

# Common academic section headings for heuristic detection
COMMON_SECTION_HEADINGS = {
    "abstract", "introduction", "background", "literature review",
    "methods", "methodology", "materials and methods", "experimental",
    "results", "findings", "analysis",
    "discussion", "conclusions", "conclusion", "summary",
    "acknowledgements", "acknowledgments", "acknowledgement",
    "references", "bibliography", "works cited",
    "appendix", "appendices", "supplementary", "supplementary material",
}

# Caption patterns for robust detection
CAPTION_PATTERNS = [
    # Figure patterns
    (r"^(figure|fig\.?)\s*(\d+)[\.:\s]*(.*)$", "figure"),
    (r"^(scheme)\s*(\d+)[\.:\s]*(.*)$", "scheme"),
    (r"^(chart)\s*(\d+)[\.:\s]*(.*)$", "chart"),
    (r"^(graph)\s*(\d+)[\.:\s]*(.*)$", "graph"),
    (r"^(plate)\s*(\d+)[\.:\s]*(.*)$", "plate"),
    (r"^(illustration)\s*(\d+)[\.:\s]*(.*)$", "illustration"),
    # Table patterns
    (r"^(table|tbl\.?)\s*(\d+)[\.:\s]*(.*)$", "table"),
    # Equation patterns
    (r"^(equation|eq\.?)\s*(\d+)[\.:\s]*(.*)$", "equation"),
    # Listing/code patterns
    (r"^(listing|code)\s*(\d+)[\.:\s]*(.*)$", "listing"),
    # Algorithm patterns
    (r"^(algorithm|alg\.?)\s*(\d+)[\.:\s]*(.*)$", "algorithm"),
]


class WordReader:
    """
    Read a DOCX file and convert it into a SciTeX writer document.

    This reader focuses on:
    - Sections (via heading styles)
    - Plain paragraphs
    - Figure/table captions (via caption style)
    - Embedded images extraction
    - References section boundary detection
    - Basic formatting (bold, italic)

    The output is a structured intermediate representation that can be
    easily fed into `scitex.writer` or exported to LaTeX/other formats.
    """

    def __init__(
        self,
        profile: BaseWordProfile,
        extract_images: bool = True,
    ):
        """
        Parameters
        ----------
        profile : BaseWordProfile
            Mapping between Word styles and SciTeX writer semantics.
        extract_images : bool
            Whether to extract embedded images from the document.
        """
        if not DOCX_AVAILABLE:
            raise ImportError(
                "python-docx is required for scitex.msword.WordReader. "
                "Install it via `pip install python-docx`."
            ) from _DOCX_IMPORT_ERROR
        self.profile = profile
        self.extract_images = extract_images

    def read(self, path: Path) -> Dict[str, Any]:
        """
        Read a DOCX file and return a SciTeX writer document.

        Parameters
        ----------
        path : Path
            Path to the DOCX file.

        Returns
        -------
        dict
            SciTeX writer document structure with keys:
            - blocks: List of document blocks
            - metadata: Profile and source information
            - images: Extracted image data (if extract_images=True)
            - references: Parsed reference entries
            - warnings: List of conversion warnings
        """
        doc = docx.Document(str(path))

        # Initialize result structure
        result: Dict[str, Any] = {
            "blocks": [],
            "metadata": {
                "profile": self.profile.name,
                "source_file": str(path),
                "import_timestamp": datetime.now().isoformat(),
            },
            "images": [],
            "references": [],
            "warnings": [],
        }

        # Extract document properties if available
        result["metadata"].update(self._extract_metadata(doc))

        # Process paragraphs and tables
        blocks = self._process_body(doc, result)
        result["blocks"] = blocks

        # Extract images
        if self.extract_images:
            result["images"] = self._extract_images(doc, path)

        # Parse references section
        result["references"] = self._parse_references(blocks)

        # Run post-import hooks
        for hook in self.profile.post_import_hooks:
            result = hook(result)

        return result

    def _extract_metadata(self, doc: DocxDocument) -> Dict[str, Any]:
        """Extract document metadata (title, author, etc.)."""
        metadata = {}
        try:
            core_props = doc.core_properties
            if core_props.title:
                metadata["title"] = core_props.title
            if core_props.author:
                metadata["author"] = core_props.author
            if core_props.subject:
                metadata["subject"] = core_props.subject
            if core_props.keywords:
                metadata["keywords"] = core_props.keywords
            if core_props.created:
                metadata["created"] = core_props.created.isoformat()
            if core_props.modified:
                metadata["modified"] = core_props.modified.isoformat()
        except Exception:
            pass  # Metadata extraction is optional
        return metadata

    def _process_body(
        self,
        doc: DocxDocument,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Process document body: paragraphs and tables."""
        blocks: List[Dict[str, Any]] = []
        in_reference_section = False
        block_index = 0

        # Build rel_id -> hash map for image detection
        rel_to_hash = {}
        if self.extract_images:
            for rel_id, rel in doc.part.rels.items():
                if "image" in rel.reltype:
                    image_bytes = rel.target_part.blob
                    image_hash = hashlib.md5(image_bytes).hexdigest()[:12]
                    rel_to_hash[rel_id] = image_hash

        # Namespace for picture detection
        pic_ns = {"pic": "http://schemas.openxmlformats.org/drawingml/2006/picture"}
        a_ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        r_ns = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}

        for element in doc.element.body:
            tag = element.tag.split("}")[-1]  # Remove namespace

            if tag == "p":
                # Process paragraph
                para = docx.text.paragraph.Paragraph(element, doc)

                # Detect inline images in this paragraph
                if self.extract_images:
                    for run in para.runs:
                        # Check for drawing elements containing pictures
                        drawings = run.element.findall(".//a:blip", namespaces=a_ns)
                        for blip in drawings:
                            embed_attr = qn("r:embed")
                            rel_id = blip.get(embed_attr)
                            if rel_id and rel_id in rel_to_hash:
                                blocks.append({
                                    "index": block_index,
                                    "type": "image",
                                    "image_hash": rel_to_hash[rel_id],
                                    "rel_id": rel_id,
                                })
                                block_index += 1

                block = self._process_paragraph(
                    para, in_reference_section, block_index
                )
                if block:
                    # Check if entering references section
                    if block["type"] == "heading" and block["text"] in (
                        self.profile.reference_section_titles
                    ):
                        in_reference_section = True
                        block["is_reference_header"] = True

                    blocks.append(block)
                    block_index += 1

            elif tag == "tbl":
                # Process table
                table = docx.table.Table(element, doc)
                block = self._process_table(table, block_index)
                blocks.append(block)
                block_index += 1

        return blocks

    def _process_paragraph(
        self,
        para,
        in_reference_section: bool,
        block_index: int,
    ) -> Optional[Dict[str, Any]]:
        """Process a single paragraph."""
        style_name = (para.style.name or "").strip() if para.style else ""
        text = para.text.strip()

        if not text:
            return None

        # Extract runs with formatting info
        runs = self._extract_runs(para)

        # Base block structure
        block: Dict[str, Any] = {
            "index": block_index,
            "text": text,
            "style": style_name,
            "runs": runs,
        }

        # Check for equations (OMML)
        equation_latex = self._extract_equation(para)
        if equation_latex:
            block["type"] = "equation"
            block["latex"] = equation_latex
            return block

        # Detect heading (style-based first, then heuristic)
        level = self._detect_heading(para, style_name, text, runs)
        if level is not None:
            block["type"] = "heading"
            block["level"] = level
            block["detection_method"] = "style" if self._heading_level_from_style(style_name) else "heuristic"
            return block

        # Detect caption (improved pattern matching)
        caption_info = self._detect_caption(style_name, text)
        if caption_info:
            block["type"] = "caption"
            block.update(caption_info)
            return block

        # Reference paragraph
        if in_reference_section:
            block["type"] = "reference-paragraph"
            ref_info = self._parse_reference_entry(text)
            block.update(ref_info)
            return block

        # List item detection
        if self._is_list_item(para):
            block["type"] = "list-item"
            list_info = self._parse_list_item(para)
            block.update(list_info)
            return block

        # Normal paragraph
        block["type"] = "paragraph"
        return block

    def _detect_heading(
        self,
        para,
        style_name: str,
        text: str,
        runs: List[Dict[str, Any]],
    ) -> Optional[int]:
        """
        Detect heading using multiple strategies:
        1. Style-based (most reliable)
        2. Font-based heuristics (bold, larger size)
        3. Content-based (known section titles)
        """
        # Strategy 1: Style-based detection
        level = self._heading_level_from_style(style_name)
        if level is not None:
            return level

        # Strategy 2: Font-based heuristics
        # Check if entire paragraph is bold and short
        text_clean = text.strip()
        if len(text_clean) < 100:  # Headings are typically short
            all_bold = all(r.get("bold") for r in runs if r.get("text", "").strip())
            if all_bold and runs:
                # Check font size - headings often larger
                avg_size = self._get_average_font_size(runs)
                if avg_size and avg_size >= 12:
                    # Check if it looks like a section heading
                    if self._looks_like_heading(text_clean):
                        return 1 if avg_size >= 14 else 2

        # Strategy 3: Content-based detection (common section titles)
        text_lower = text_clean.lower().rstrip(".:;")
        # Check numbered sections: "1. Introduction", "2.1 Methods"
        numbered_match = re.match(r"^(\d+(?:\.\d+)*)[\.:\s]+(.+)$", text_clean)
        if numbered_match:
            section_text = numbered_match.group(2).lower().strip()
            if section_text in COMMON_SECTION_HEADINGS:
                depth = numbered_match.group(1).count(".")
                return min(depth + 1, 4)

        # Check unnumbered common headings (if bold or all caps)
        if text_lower in COMMON_SECTION_HEADINGS:
            is_bold = all(r.get("bold") for r in runs if r.get("text", "").strip())
            is_all_caps = text_clean.isupper() and len(text_clean) > 3
            if is_bold or is_all_caps:
                return 1

        return None

    def _looks_like_heading(self, text: str) -> bool:
        """Check if text looks like a heading based on content patterns."""
        text_lower = text.lower().rstrip(".:;")

        # Check common section headings
        if text_lower in COMMON_SECTION_HEADINGS:
            return True

        # Check numbered sections
        if re.match(r"^\d+(?:\.\d+)*\s+\w", text):
            return True

        # All caps short text
        if text.isupper() and 3 < len(text) < 50:
            return True

        return False

    def _get_average_font_size(self, runs: List[Dict[str, Any]]) -> Optional[float]:
        """Get average font size from runs."""
        sizes = [r["font_size"] for r in runs if r.get("font_size")]
        return sum(sizes) / len(sizes) if sizes else None

    def _detect_caption(self, style_name: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Detect and parse captions using multiple patterns.
        Returns caption info dict or None.
        """
        # Check by style first
        if style_name == self.profile.caption_style:
            return self._parse_caption(text)

        # Check using comprehensive patterns
        text_stripped = text.strip()
        for pattern, caption_type in CAPTION_PATTERNS:
            match = re.match(pattern, text_stripped, re.IGNORECASE)
            if match:
                return {
                    "caption_type": caption_type,
                    "number": int(match.group(2)),
                    "caption_text": match.group(3).strip(),
                }

        # Check profile-specific prefixes
        if self._is_caption(style_name, text):
            return self._parse_caption(text)

        return None

    def _extract_equation(self, para) -> Optional[str]:
        """
        Extract equation from paragraph if it contains OMML (Office Math Markup).
        Returns LaTeX representation or None.
        """
        try:
            # Check for oMath elements
            omml_ns = {"m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
            math_elements = para._element.findall(".//m:oMath", namespaces=omml_ns)

            if not math_elements:
                return None

            # Basic OMML to LaTeX conversion
            latex_parts = []
            for math_elem in math_elements:
                latex = self._omml_to_latex(math_elem)
                if latex:
                    latex_parts.append(latex)

            return " ".join(latex_parts) if latex_parts else None
        except Exception:
            return None

    def _omml_to_latex(self, math_elem) -> str:
        """
        Convert OMML element to LaTeX string.
        This is a basic converter - handles common cases.
        """
        omml_ns = {"m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}

        def get_text(elem) -> str:
            """Recursively get text from element."""
            texts = []
            if elem.text:
                texts.append(elem.text)
            for child in elem:
                texts.append(get_text(child))
                if child.tail:
                    texts.append(child.tail)
            return "".join(texts)

        def convert_element(elem) -> str:
            """Convert a single OMML element to LaTeX."""
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

            if tag == "r":  # Run (text)
                return get_text(elem)
            elif tag == "f":  # Fraction
                num = elem.find("m:num", namespaces=omml_ns)
                den = elem.find("m:den", namespaces=omml_ns)
                num_tex = convert_children(num) if num is not None else ""
                den_tex = convert_children(den) if den is not None else ""
                return f"\\frac{{{num_tex}}}{{{den_tex}}}"
            elif tag == "rad":  # Radical/root
                deg = elem.find("m:deg", namespaces=omml_ns)
                content = elem.find("m:e", namespaces=omml_ns)
                content_tex = convert_children(content) if content is not None else ""
                if deg is not None and get_text(deg).strip():
                    deg_tex = convert_children(deg)
                    return f"\\sqrt[{deg_tex}]{{{content_tex}}}"
                return f"\\sqrt{{{content_tex}}}"
            elif tag == "sSup":  # Superscript
                base = elem.find("m:e", namespaces=omml_ns)
                sup = elem.find("m:sup", namespaces=omml_ns)
                base_tex = convert_children(base) if base is not None else ""
                sup_tex = convert_children(sup) if sup is not None else ""
                return f"{base_tex}^{{{sup_tex}}}"
            elif tag == "sSub":  # Subscript
                base = elem.find("m:e", namespaces=omml_ns)
                sub = elem.find("m:sub", namespaces=omml_ns)
                base_tex = convert_children(base) if base is not None else ""
                sub_tex = convert_children(sub) if sub is not None else ""
                return f"{base_tex}_{{{sub_tex}}}"
            elif tag == "sSubSup":  # Sub-superscript
                base = elem.find("m:e", namespaces=omml_ns)
                sub = elem.find("m:sub", namespaces=omml_ns)
                sup = elem.find("m:sup", namespaces=omml_ns)
                base_tex = convert_children(base) if base is not None else ""
                sub_tex = convert_children(sub) if sub is not None else ""
                sup_tex = convert_children(sup) if sup is not None else ""
                return f"{base_tex}_{{{sub_tex}}}^{{{sup_tex}}}"
            elif tag == "nary":  # N-ary (sum, product, integral)
                chr_elem = elem.find(".//m:chr", namespaces=omml_ns)
                symbol = chr_elem.get(qn("m:val")) if chr_elem is not None else "∑"
                symbol_map = {"∑": "\\sum", "∏": "\\prod", "∫": "\\int", "∮": "\\oint"}
                latex_sym = symbol_map.get(symbol, symbol)
                sub = elem.find("m:sub", namespaces=omml_ns)
                sup = elem.find("m:sup", namespaces=omml_ns)
                content = elem.find("m:e", namespaces=omml_ns)
                result = latex_sym
                if sub is not None:
                    result += f"_{{{convert_children(sub)}}}"
                if sup is not None:
                    result += f"^{{{convert_children(sup)}}}"
                if content is not None:
                    result += f" {convert_children(content)}"
                return result
            elif tag == "d":  # Delimiter (parentheses, brackets)
                content = elem.find("m:e", namespaces=omml_ns)
                content_tex = convert_children(content) if content is not None else ""
                beg = elem.find(".//m:begChr", namespaces=omml_ns)
                end = elem.find(".//m:endChr", namespaces=omml_ns)
                left = beg.get(qn("m:val")) if beg is not None else "("
                right = end.get(qn("m:val")) if end is not None else ")"
                return f"\\left{left}{content_tex}\\right{right}"
            elif tag in ("e", "num", "den", "sub", "sup", "deg"):
                # Container elements - just process children
                return convert_children(elem)
            else:
                # Unknown element - try to get text
                return convert_children(elem)

        def convert_children(elem) -> str:
            """Convert all children of an element."""
            if elem is None:
                return ""
            parts = []
            for child in elem:
                parts.append(convert_element(child))
            return "".join(parts)

        return convert_element(math_elem)

    def _is_list_item(self, para) -> bool:
        """Check if paragraph is a list item."""
        try:
            # Check for numbering properties
            pPr = para._element.find(qn("w:pPr"))
            if pPr is not None:
                numPr = pPr.find(qn("w:numPr"))
                if numPr is not None:
                    return True

            # Check for bullet/number at start of text
            text = para.text.strip()
            if re.match(r"^[\u2022\u2023\u25E6\u2043\u2219•‣◦⁃∙]\s", text):
                return True
            if re.match(r"^(\d+[\.\):]|\([a-z]\)|\([ivxlc]+\)|[a-z][\.\)])\s", text, re.IGNORECASE):
                return True

            return False
        except Exception:
            return False

    def _parse_list_item(self, para) -> Dict[str, Any]:
        """Parse list item to extract level and content."""
        info: Dict[str, Any] = {"list_type": "unordered", "level": 0}

        try:
            pPr = para._element.find(qn("w:pPr"))
            if pPr is not None:
                numPr = pPr.find(qn("w:numPr"))
                if numPr is not None:
                    ilvl = numPr.find(qn("w:ilvl"))
                    if ilvl is not None:
                        info["level"] = int(ilvl.get(qn("w:val"), 0))

            # Detect ordered vs unordered
            text = para.text.strip()
            if re.match(r"^\d+[\.\):]\s", text):
                info["list_type"] = "ordered"
        except Exception:
            pass

        return info

    def _extract_runs(self, para) -> List[Dict[str, Any]]:
        """Extract formatted runs from a paragraph."""
        runs = []
        for run in para.runs:
            if not run.text:
                continue
            run_data = {
                "text": run.text,
                "bold": run.bold,
                "italic": run.italic,
                "underline": run.underline is not None,
            }
            if run.font.size:
                run_data["font_size"] = run.font.size.pt
            if run.font.name:
                run_data["font_name"] = run.font.name
            runs.append(run_data)
        return runs

    def _heading_level_from_style(self, style_name: str) -> Optional[int]:
        """Return heading level for a given Word style, or None."""
        for level, expected_style in self.profile.heading_styles.items():
            if style_name == expected_style:
                return level
        return None

    def _is_caption(self, style_name: str, text: str) -> bool:
        """Check if paragraph is a caption."""
        if style_name == self.profile.caption_style:
            return True

        # Check by prefix
        text_lower = text.lower()
        prefixes = (
            self.profile.figure_caption_prefixes
            + self.profile.table_caption_prefixes
        )
        for prefix in prefixes:
            if text_lower.startswith(prefix.lower()):
                return True
        return False

    def _parse_caption(self, text: str) -> Dict[str, Any]:
        """Parse caption text to extract figure/table number."""
        info: Dict[str, Any] = {}

        # Check figure
        for prefix in self.profile.figure_caption_prefixes:
            pattern = rf"^{re.escape(prefix)}\.?\s*(\d+)[\.:]?\s*(.*)$"
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                info["caption_type"] = "figure"
                info["number"] = int(match.group(1))
                info["caption_text"] = match.group(2).strip()
                return info

        # Check table
        for prefix in self.profile.table_caption_prefixes:
            pattern = rf"^{re.escape(prefix)}\.?\s*(\d+)[\.:]?\s*(.*)$"
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                info["caption_type"] = "table"
                info["number"] = int(match.group(1))
                info["caption_text"] = match.group(2).strip()
                return info

        info["caption_type"] = "unknown"
        info["caption_text"] = text
        return info

    def _parse_reference_entry(self, text: str) -> Dict[str, Any]:
        """Parse a reference entry to extract citation number."""
        info: Dict[str, Any] = {}

        # Try to extract numbered reference: [1], 1., (1), etc.
        patterns = [
            r"^\[(\d+)\]",  # [1] Author...
            r"^(\d+)\.",  # 1. Author...
            r"^\((\d+)\)",  # (1) Author...
        ]
        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                info["ref_number"] = int(match.group(1))
                info["ref_text"] = re.sub(pattern, "", text).strip()
                break
        else:
            info["ref_text"] = text

        return info

    def _process_table(
        self,
        table,
        block_index: int,
    ) -> Dict[str, Any]:
        """Process a table."""
        rows = []
        for row in table.rows:
            cells = []
            for cell in row.cells:
                cells.append(cell.text.strip())
            rows.append(cells)

        return {
            "index": block_index,
            "type": "table",
            "rows": rows,
            "num_rows": len(rows),
            "num_cols": len(rows[0]) if rows else 0,
        }

    def _extract_images(
        self,
        doc: DocxDocument,
        source_path: Path,
    ) -> List[Dict[str, Any]]:
        """Extract embedded images from the document."""
        images = []

        try:
            for rel_id, rel in doc.part.rels.items():
                if "image" in rel.reltype:
                    image_part = rel.target_part
                    image_bytes = image_part.blob

                    # Generate hash for deduplication
                    image_hash = hashlib.md5(image_bytes).hexdigest()[:12]

                    # Determine extension from content type
                    content_type = image_part.content_type
                    ext_map = {
                        "image/png": ".png",
                        "image/jpeg": ".jpg",
                        "image/gif": ".gif",
                        "image/tiff": ".tiff",
                        "image/bmp": ".bmp",
                    }
                    ext = ext_map.get(content_type, ".png")

                    images.append(
                        {
                            "rel_id": rel_id,
                            "hash": image_hash,
                            "content_type": content_type,
                            "extension": ext,
                            "size_bytes": len(image_bytes),
                            "data": image_bytes,  # Raw bytes
                        }
                    )
        except Exception as e:
            pass  # Image extraction is optional

        return images

    def _parse_references(
        self,
        blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract and structure references from blocks."""
        references = []
        for block in blocks:
            if block.get("type") == "reference-paragraph":
                ref_entry = {
                    "number": block.get("ref_number"),
                    "text": block.get("ref_text", block.get("text", "")),
                    "raw": block.get("text", ""),
                }
                references.append(ref_entry)
        return references


__all__ = ["WordReader"]
