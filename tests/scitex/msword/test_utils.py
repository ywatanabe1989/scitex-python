#!/usr/bin/env python3
# Timestamp: 2026-01-05 14:00:00
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/msword/test_utils.py

"""Tests for scitex.msword.utils module."""

import pytest


class TestLinkCaptionsToImages:
    """Tests for link_captions_to_images function."""

    def test_link_captions_to_images_basic(self):
        """Should link figure captions to images by number."""
        from scitex.msword.utils import link_captions_to_images

        doc = {
            "blocks": [
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 1,
                    "caption_text": "First figure",
                },
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 2,
                    "caption_text": "Second figure",
                },
            ],
            "images": [
                {"hash": "hash_img_1"},
                {"hash": "hash_img_2"},
            ],
        }

        result = link_captions_to_images(doc)

        assert result["blocks"][0]["image_hash"] == "hash_img_1"
        assert result["blocks"][1]["image_hash"] == "hash_img_2"

    def test_link_captions_to_images_with_more_images(self):
        """Should handle more images than captions."""
        from scitex.msword.utils import link_captions_to_images

        doc = {
            "blocks": [
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 1,
                    "caption_text": "Only figure",
                },
            ],
            "images": [
                {"hash": "hash_1"},
                {"hash": "hash_2"},
                {"hash": "hash_3"},
            ],
        }

        result = link_captions_to_images(doc)

        assert result["blocks"][0]["image_hash"] == "hash_1"

    def test_link_captions_to_images_with_more_captions(self):
        """Should handle more captions than images gracefully."""
        from scitex.msword.utils import link_captions_to_images

        doc = {
            "blocks": [
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 1,
                    "caption_text": "First",
                },
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 2,
                    "caption_text": "Second",
                },
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 3,
                    "caption_text": "Third",
                },
            ],
            "images": [
                {"hash": "hash_only"},
            ],
        }

        result = link_captions_to_images(doc)

        assert result["blocks"][0]["image_hash"] == "hash_only"
        assert "image_hash" not in result["blocks"][1]
        assert "image_hash" not in result["blocks"][2]

    def test_link_captions_to_images_empty_images(self):
        """Should handle empty images list."""
        from scitex.msword.utils import link_captions_to_images

        doc = {
            "blocks": [
                {"type": "caption", "caption_type": "figure", "number": 1},
            ],
            "images": [],
        }

        result = link_captions_to_images(doc)

        assert "image_hash" not in result["blocks"][0]

    def test_link_captions_to_images_no_figure_captions(self):
        """Should handle documents with no figure captions."""
        from scitex.msword.utils import link_captions_to_images

        doc = {
            "blocks": [
                {"type": "caption", "caption_type": "table", "number": 1},
                {"type": "paragraph", "text": "Some text"},
            ],
            "images": [{"hash": "hash_1"}],
        }

        result = link_captions_to_images(doc)

        # Table captions should not be linked
        assert "image_hash" not in result["blocks"][0]

    def test_link_captions_to_images_mixed_blocks(self):
        """Should correctly link captions in mixed block types."""
        from scitex.msword.utils import link_captions_to_images

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Figures"},
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 1,
                    "caption_text": "Fig 1",
                },
                {"type": "paragraph", "text": "Description"},
                {
                    "type": "caption",
                    "caption_type": "table",
                    "number": 1,
                    "caption_text": "Table 1",
                },
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 2,
                    "caption_text": "Fig 2",
                },
            ],
            "images": [
                {"hash": "img_hash_1"},
                {"hash": "img_hash_2"},
            ],
        }

        result = link_captions_to_images(doc)

        # Only figure captions should have image_hash
        assert result["blocks"][1]["image_hash"] == "img_hash_1"
        assert "image_hash" not in result["blocks"][3]  # table caption
        assert result["blocks"][4]["image_hash"] == "img_hash_2"

    def test_link_captions_to_images_non_sequential_numbers(self):
        """Should link by figure number, not block order."""
        from scitex.msword.utils import link_captions_to_images

        doc = {
            "blocks": [
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 2,
                    "caption_text": "Second",
                },
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 1,
                    "caption_text": "First",
                },
            ],
            "images": [
                {"hash": "hash_0"},
                {"hash": "hash_1"},
            ],
        }

        result = link_captions_to_images(doc)

        # Figure 2 -> images[1], Figure 1 -> images[0]
        assert result["blocks"][0]["image_hash"] == "hash_1"  # Figure 2
        assert result["blocks"][1]["image_hash"] == "hash_0"  # Figure 1


class TestLinkCaptionsToImagesByProximity:
    """Tests for link_captions_to_images_by_proximity function."""

    def test_link_by_proximity_basic(self):
        """Should link captions to nearest preceding images."""
        from scitex.msword.utils import link_captions_to_images_by_proximity

        doc = {
            "blocks": [
                {"type": "image", "image_hash": "img_1"},
                {"type": "caption", "caption_type": "figure", "number": 1},
                {"type": "image", "image_hash": "img_2"},
                {"type": "caption", "caption_type": "figure", "number": 2},
            ],
            "images": [],
        }

        result = link_captions_to_images_by_proximity(doc)

        assert result["blocks"][1]["image_hash"] == "img_1"
        assert result["blocks"][3]["image_hash"] == "img_2"

    def test_link_by_proximity_fallback_to_images_list(self):
        """Should fall back to images list when no image blocks."""
        from scitex.msword.utils import link_captions_to_images_by_proximity

        doc = {
            "blocks": [
                {"type": "caption", "caption_type": "figure", "number": 1},
                {"type": "caption", "caption_type": "figure", "number": 2},
            ],
            "images": [
                {"hash": "fallback_1"},
                {"hash": "fallback_2"},
            ],
        }

        result = link_captions_to_images_by_proximity(doc)

        assert result["blocks"][0]["image_hash"] == "fallback_1"
        assert result["blocks"][1]["image_hash"] == "fallback_2"

    def test_link_by_proximity_no_images_at_all(self):
        """Should handle case with no images."""
        from scitex.msword.utils import link_captions_to_images_by_proximity

        doc = {
            "blocks": [
                {"type": "caption", "caption_type": "figure", "number": 1},
            ],
            "images": [],
        }

        result = link_captions_to_images_by_proximity(doc)

        assert "image_hash" not in result["blocks"][0]

    def test_link_by_proximity_prefers_preceding_image(self):
        """Should prefer preceding images over following images."""
        from scitex.msword.utils import link_captions_to_images_by_proximity

        doc = {
            "blocks": [
                {"type": "image", "image_hash": "before_img"},
                {"type": "paragraph", "text": "text"},
                {"type": "caption", "caption_type": "figure", "number": 1},
                {"type": "paragraph", "text": "more text"},
                {"type": "image", "image_hash": "after_img"},
            ],
            "images": [],
        }

        result = link_captions_to_images_by_proximity(doc)

        # Caption at index 2, image at index 0 is closer than image at index 4
        assert result["blocks"][2]["image_hash"] == "before_img"

    def test_link_by_proximity_uses_following_if_no_preceding(self):
        """Should use following image if no preceding image available."""
        from scitex.msword.utils import link_captions_to_images_by_proximity

        doc = {
            "blocks": [
                {"type": "caption", "caption_type": "figure", "number": 1},
                {"type": "paragraph", "text": "text"},
                {"type": "image", "image_hash": "only_img"},
            ],
            "images": [],
        }

        result = link_captions_to_images_by_proximity(doc)

        assert result["blocks"][0]["image_hash"] == "only_img"

    def test_link_by_proximity_avoids_reusing_images(self):
        """Should not reuse already linked images."""
        from scitex.msword.utils import link_captions_to_images_by_proximity

        doc = {
            "blocks": [
                {"type": "image", "image_hash": "shared_img"},
                {"type": "caption", "caption_type": "figure", "number": 1},
                {"type": "caption", "caption_type": "figure", "number": 2},
                {"type": "image", "image_hash": "second_img"},
            ],
            "images": [],
        }

        result = link_captions_to_images_by_proximity(doc)

        # First caption gets nearest (shared_img)
        assert result["blocks"][1]["image_hash"] == "shared_img"
        # Second caption gets the remaining one (second_img)
        assert result["blocks"][2]["image_hash"] == "second_img"


class TestNormalizeSectionHeadings:
    """Tests for normalize_section_headings function."""

    def test_normalize_intro(self):
        """Should normalize 'intro' to 'Introduction'."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "intro"},
            ]
        }

        result = normalize_section_headings(doc)

        assert result["blocks"][0]["text"] == "Introduction"

    def test_normalize_introduction(self):
        """Should normalize 'introduction' to 'Introduction'."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "introduction"},
            ]
        }

        result = normalize_section_headings(doc)

        assert result["blocks"][0]["text"] == "Introduction"

    def test_normalize_methods(self):
        """Should normalize 'method' to 'Methods'."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "method"},
            ]
        }

        result = normalize_section_headings(doc)

        assert result["blocks"][0]["text"] == "Methods"

    def test_normalize_results(self):
        """Should normalize 'result' to 'Results'."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "result"},
            ]
        }

        result = normalize_section_headings(doc)

        assert result["blocks"][0]["text"] == "Results"

    def test_normalize_conclusions(self):
        """Should normalize 'conclusion' to 'Conclusions'."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "conclusion"},
            ]
        }

        result = normalize_section_headings(doc)

        assert result["blocks"][0]["text"] == "Conclusions"

    def test_normalize_acknowledgements(self):
        """Should normalize 'acknowledgement' to 'Acknowledgements'."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "acknowledgement"},
            ]
        }

        result = normalize_section_headings(doc)

        assert result["blocks"][0]["text"] == "Acknowledgements"

    def test_normalize_bibliography_to_references(self):
        """Should normalize 'bibliography' to 'References'."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "bibliography"},
            ]
        }

        result = normalize_section_headings(doc)

        assert result["blocks"][0]["text"] == "References"

    def test_normalize_only_level1_headings(self):
        """Should only normalize level 1 headings."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "heading", "level": 2, "text": "intro"},
                {"type": "heading", "level": 3, "text": "method"},
            ]
        }

        result = normalize_section_headings(doc)

        # Level 2 and 3 headings should not be normalized
        assert result["blocks"][0]["text"] == "intro"
        assert result["blocks"][1]["text"] == "method"

    def test_normalize_case_insensitive(self):
        """Should handle different cases."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "INTRODUCTION"},
                {"type": "heading", "level": 1, "text": "Methods"},
            ]
        }

        result = normalize_section_headings(doc)

        # Both should be normalized
        assert result["blocks"][0]["text"] == "Introduction"
        assert result["blocks"][1]["text"] == "Methods"

    def test_normalize_preserves_other_blocks(self):
        """Should preserve non-heading blocks."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "paragraph", "text": "intro"},
                {"type": "caption", "text": "method"},
            ]
        }

        result = normalize_section_headings(doc)

        # Paragraphs and captions should not be changed
        assert result["blocks"][0]["text"] == "intro"
        assert result["blocks"][1]["text"] == "method"

    def test_normalize_materials_and_methods(self):
        """Should normalize 'materials and methods'."""
        from scitex.msword.utils import normalize_section_headings

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "materials and methods"},
            ]
        }

        result = normalize_section_headings(doc)

        assert result["blocks"][0]["text"] == "Materials and Methods"


class TestValidateDocument:
    """Tests for validate_document function."""

    def test_validate_complete_document(self):
        """Should not add warnings for complete document."""
        from scitex.msword.utils import validate_document

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Introduction"},
                {"type": "heading", "level": 1, "text": "Methods"},
                {"type": "heading", "level": 1, "text": "Results"},
                {"type": "heading", "level": 1, "text": "Discussion"},
                {"type": "heading", "level": 1, "text": "References"},
            ],
            "references": [{"number": 1, "text": "Ref 1"}],
            "warnings": [],
        }

        result = validate_document(doc)

        assert len(result["warnings"]) == 0

    def test_validate_missing_introduction(self):
        """Should warn about missing Introduction."""
        from scitex.msword.utils import validate_document

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Methods"},
                {"type": "heading", "level": 1, "text": "Results"},
                {"type": "heading", "level": 1, "text": "Discussion"},
                {"type": "heading", "level": 1, "text": "References"},
            ],
            "references": [{"number": 1, "text": "Ref"}],
            "warnings": [],
        }

        result = validate_document(doc)

        assert any("Introduction" in w for w in result["warnings"])

    def test_validate_missing_methods(self):
        """Should warn about missing Methods."""
        from scitex.msword.utils import validate_document

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Introduction"},
                {"type": "heading", "level": 1, "text": "Results"},
                {"type": "heading", "level": 1, "text": "Discussion"},
                {"type": "heading", "level": 1, "text": "References"},
            ],
            "references": [{"number": 1, "text": "Ref"}],
            "warnings": [],
        }

        result = validate_document(doc)

        assert any("Methods" in w for w in result["warnings"])

    def test_validate_missing_results(self):
        """Should warn about missing Results."""
        from scitex.msword.utils import validate_document

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Introduction"},
                {"type": "heading", "level": 1, "text": "Methods"},
                {"type": "heading", "level": 1, "text": "Discussion"},
                {"type": "heading", "level": 1, "text": "References"},
            ],
            "references": [{"number": 1, "text": "Ref"}],
            "warnings": [],
        }

        result = validate_document(doc)

        assert any("Results" in w for w in result["warnings"])

    def test_validate_missing_discussion(self):
        """Should warn about missing Discussion."""
        from scitex.msword.utils import validate_document

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Introduction"},
                {"type": "heading", "level": 1, "text": "Methods"},
                {"type": "heading", "level": 1, "text": "Results"},
                {"type": "heading", "level": 1, "text": "References"},
            ],
            "references": [{"number": 1, "text": "Ref"}],
            "warnings": [],
        }

        result = validate_document(doc)

        assert any("Discussion" in w for w in result["warnings"])

    def test_validate_missing_references_section(self):
        """Should warn about missing References section."""
        from scitex.msword.utils import validate_document

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Introduction"},
                {"type": "heading", "level": 1, "text": "Methods"},
                {"type": "heading", "level": 1, "text": "Results"},
                {"type": "heading", "level": 1, "text": "Discussion"},
            ],
            "references": [{"number": 1, "text": "Ref"}],
            "warnings": [],
        }

        result = validate_document(doc)

        assert any("References" in w for w in result["warnings"])

    def test_validate_duplicate_figure_numbers(self):
        """Should warn about duplicate figure numbers."""
        from scitex.msword.utils import validate_document

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Introduction"},
                {"type": "heading", "level": 1, "text": "Methods"},
                {"type": "heading", "level": 1, "text": "Results"},
                {"type": "heading", "level": 1, "text": "Discussion"},
                {"type": "heading", "level": 1, "text": "References"},
                {"type": "caption", "caption_type": "figure", "number": 1},
                {
                    "type": "caption",
                    "caption_type": "figure",
                    "number": 1,
                },  # Duplicate!
            ],
            "references": [{"number": 1, "text": "Ref"}],
            "warnings": [],
        }

        result = validate_document(doc)

        assert any("Duplicate figure number: 1" in w for w in result["warnings"])

    def test_validate_no_references(self):
        """Should warn about missing references."""
        from scitex.msword.utils import validate_document

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Introduction"},
                {"type": "heading", "level": 1, "text": "Methods"},
                {"type": "heading", "level": 1, "text": "Results"},
                {"type": "heading", "level": 1, "text": "Discussion"},
                {"type": "heading", "level": 1, "text": "References"},
            ],
            "references": [],
            "warnings": [],
        }

        result = validate_document(doc)

        assert any("No references found" in w for w in result["warnings"])

    def test_validate_reference_paragraphs_count_as_references(self):
        """Should not warn if reference-paragraphs exist."""
        from scitex.msword.utils import validate_document

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Introduction"},
                {"type": "heading", "level": 1, "text": "Methods"},
                {"type": "heading", "level": 1, "text": "Results"},
                {"type": "heading", "level": 1, "text": "Discussion"},
                {"type": "heading", "level": 1, "text": "References"},
                {"type": "reference-paragraph", "ref_number": 1, "text": "Ref 1"},
            ],
            "references": [],  # Empty list but we have reference-paragraph blocks
            "warnings": [],
        }

        result = validate_document(doc)

        assert not any("No references found" in w for w in result["warnings"])

    def test_validate_preserves_existing_warnings(self):
        """Should preserve existing warnings."""
        from scitex.msword.utils import validate_document

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "Introduction"},
                {"type": "heading", "level": 1, "text": "Methods"},
                {"type": "heading", "level": 1, "text": "Results"},
                {"type": "heading", "level": 1, "text": "Discussion"},
                {"type": "heading", "level": 1, "text": "References"},
            ],
            "references": [{"number": 1, "text": "Ref"}],
            "warnings": ["Existing warning"],
        }

        result = validate_document(doc)

        assert "Existing warning" in result["warnings"]


class TestCreatePostImportHook:
    """Tests for create_post_import_hook function."""

    def test_create_hook_single_function(self):
        """Should create hook from single function."""
        from scitex.msword.utils import create_post_import_hook

        def add_marker(doc):
            doc["marker"] = True
            return doc

        hook = create_post_import_hook(add_marker)
        result = hook({"blocks": []})

        assert result["marker"] is True

    def test_create_hook_multiple_functions(self):
        """Should chain multiple functions."""
        from scitex.msword.utils import create_post_import_hook

        def add_first(doc):
            doc["first"] = True
            return doc

        def add_second(doc):
            doc["second"] = True
            return doc

        hook = create_post_import_hook(add_first, add_second)
        result = hook({"blocks": []})

        assert result["first"] is True
        assert result["second"] is True

    def test_create_hook_order_preserved(self):
        """Should apply functions in order."""
        from scitex.msword.utils import create_post_import_hook

        def append_a(doc):
            doc["order"] = doc.get("order", "") + "A"
            return doc

        def append_b(doc):
            doc["order"] = doc.get("order", "") + "B"
            return doc

        hook = create_post_import_hook(append_a, append_b)
        result = hook({"blocks": []})

        assert result["order"] == "AB"

    def test_create_hook_with_real_utils(self):
        """Should work with actual utility functions."""
        from scitex.msword.utils import (
            create_post_import_hook,
            normalize_section_headings,
            validate_document,
        )

        hook = create_post_import_hook(normalize_section_headings, validate_document)

        doc = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "intro"},
                {"type": "heading", "level": 1, "text": "method"},
            ],
            "references": [],
            "warnings": [],
        }

        result = hook(doc)

        # Check normalize_section_headings was applied
        assert result["blocks"][0]["text"] == "Introduction"
        assert result["blocks"][1]["text"] == "Methods"
        # Check validate_document was applied
        assert "warnings" in result

    def test_create_hook_empty_functions(self):
        """Should handle no functions gracefully."""
        from scitex.msword.utils import create_post_import_hook

        hook = create_post_import_hook()
        doc = {"blocks": [], "test": "value"}
        result = hook(doc)

        assert result["test"] == "value"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/msword/utils.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: 2025-12-11 16:45:00
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/msword/utils.py
# 
# """
# Utility functions for processing MS Word documents.
# 
# These functions can be used as post_import_hooks or called directly
# to process document structures.
# """
# 
# from __future__ import annotations
# 
# from typing import Any, Dict, List
# 
# 
# def link_captions_to_images(doc: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Link figure captions to images by matching order.
# 
#     This function pairs figure captions with images based on their
#     sequential order in the document. Each figure caption is assigned
#     an `image_hash` that corresponds to the image at the same position.
# 
#     Parameters
#     ----------
#     doc : dict
#         SciTeX writer document with 'blocks' and 'images' keys.
# 
#     Returns
#     -------
#     dict
#         The same document with image_hash added to figure captions.
# 
#     Examples
#     --------
#     >>> from scitex.msword import load_docx
#     >>> from scitex.msword.utils import link_captions_to_images
#     >>> doc = load_docx("manuscript.docx")
#     >>> doc = link_captions_to_images(doc)
#     >>> # Now captions have image_hash for LaTeX export
#     """
#     blocks = doc.get("blocks", [])
#     images = doc.get("images", [])
# 
#     # Find all figure captions
#     figure_captions = [
#         b for b in blocks
#         if b.get("type") == "caption" and b.get("caption_type") == "figure"
#     ]
# 
#     # Link by order (figure 1 -> image 0, figure 2 -> image 1, etc.)
#     for caption in figure_captions:
#         fig_num = caption.get("number")
#         if fig_num is not None and isinstance(fig_num, int):
#             # Figure numbers are typically 1-indexed
#             img_idx = fig_num - 1
#             if 0 <= img_idx < len(images):
#                 caption["image_hash"] = images[img_idx].get("hash")
# 
#     return doc
# 
# 
# def link_captions_to_images_by_proximity(doc: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Link figure captions to images by document proximity.
# 
#     This function uses the image blocks (type="image") that are inserted
#     at their actual positions in the document body. It finds the nearest
#     unlinked image block to each figure caption.
# 
#     Parameters
#     ----------
#     doc : dict
#         SciTeX writer document.
# 
#     Returns
#     -------
#     dict
#         Document with image_hash added to captions.
#     """
#     blocks = doc.get("blocks", [])
# 
#     # Collect image blocks and figure captions with their indices
#     image_blocks = []
#     figure_captions = []
# 
#     for i, block in enumerate(blocks):
#         if block.get("type") == "image":
#             image_blocks.append((i, block))
#         elif block.get("type") == "caption" and block.get("caption_type") == "figure":
#             figure_captions.append((i, block))
# 
#     if not image_blocks:
#         # Fallback to old behavior using doc["images"] list
#         images = doc.get("images", [])
#         if not images:
#             return doc
#         image_hashes = [img.get("hash") for img in images]
#         for idx, (_, caption) in enumerate(figure_captions):
#             if idx < len(image_hashes):
#                 caption["image_hash"] = image_hashes[idx]
#         return doc
# 
#     used_images = set()
# 
#     # For each caption, find the nearest preceding image block
#     for cap_idx, caption in figure_captions:
#         best_img_idx = None
#         best_img_hash = None
#         best_distance = float("inf")
# 
#         for img_idx, img_block in image_blocks:
#             img_hash = img_block.get("image_hash")
#             if img_hash in used_images:
#                 continue
# 
#             # Prefer images that come before the caption (typical layout)
#             distance = cap_idx - img_idx
#             if distance >= 0 and distance < best_distance:
#                 best_distance = distance
#                 best_img_idx = img_idx
#                 best_img_hash = img_hash
# 
#         # If no preceding image, try following images
#         if best_img_hash is None:
#             for img_idx, img_block in image_blocks:
#                 img_hash = img_block.get("image_hash")
#                 if img_hash in used_images:
#                     continue
# 
#                 distance = abs(cap_idx - img_idx)
#                 if distance < best_distance:
#                     best_distance = distance
#                     best_img_idx = img_idx
#                     best_img_hash = img_hash
# 
#         if best_img_hash:
#             caption["image_hash"] = best_img_hash
#             used_images.add(best_img_hash)
# 
#     return doc
# 
# 
# def normalize_section_headings(doc: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Normalize section headings for consistency.
# 
#     Converts common section titles to standard academic format:
#     - "intro" -> "Introduction"
#     - "method" -> "Methods"
#     - etc.
# 
#     Parameters
#     ----------
#     doc : dict
#         SciTeX writer document.
# 
#     Returns
#     -------
#     dict
#         Document with normalized headings.
#     """
#     blocks = doc.get("blocks", [])
# 
#     # Common normalizations
#     normalizations = {
#         "intro": "Introduction",
#         "introduction": "Introduction",
#         "method": "Methods",
#         "methods": "Methods",
#         "materials and methods": "Materials and Methods",
#         "result": "Results",
#         "results": "Results",
#         "discussion": "Discussion",
#         "conclusion": "Conclusions",
#         "conclusions": "Conclusions",
#         "acknowledgement": "Acknowledgements",
#         "acknowledgements": "Acknowledgements",
#         "reference": "References",
#         "references": "References",
#         "bibliography": "References",
#     }
# 
#     for block in blocks:
#         if block.get("type") == "heading" and block.get("level") == 1:
#             text = block.get("text", "").strip().lower()
#             if text in normalizations:
#                 block["text"] = normalizations[text]
# 
#     return doc
# 
# 
# def validate_document(doc: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Validate document structure and add warnings.
# 
#     Checks for common issues:
#     - Missing required sections
#     - Unmatched caption numbers
#     - Empty references section
#     - Duplicate figure numbers
# 
#     Parameters
#     ----------
#     doc : dict
#         SciTeX writer document.
# 
#     Returns
#     -------
#     dict
#         Document with warnings added.
#     """
#     blocks = doc.get("blocks", [])
#     warnings = doc.get("warnings", [])
# 
#     # Check for required sections
#     headings = [b.get("text", "").lower() for b in blocks if b.get("type") == "heading"]
# 
#     required_sections = ["introduction", "methods", "results", "discussion", "references"]
#     for section in required_sections:
#         if not any(section in h for h in headings):
#             warnings.append(f"Missing section: {section.title()}")
# 
#     # Check for duplicate figure numbers
#     figure_numbers = [
#         b.get("number") for b in blocks
#         if b.get("type") == "caption" and b.get("caption_type") == "figure"
#     ]
#     seen = set()
#     for num in figure_numbers:
#         if num in seen:
#             warnings.append(f"Duplicate figure number: {num}")
#         seen.add(num)
# 
#     # Check for missing references
#     references = doc.get("references", [])
#     if not references:
#         ref_blocks = [b for b in blocks if b.get("type") == "reference-paragraph"]
#         if not ref_blocks:
#             warnings.append("No references found in document")
# 
#     doc["warnings"] = warnings
#     return doc
# 
# 
# def create_post_import_hook(*functions):
#     """
#     Create a composite post_import_hook from multiple functions.
# 
#     Parameters
#     ----------
#     *functions : callable
#         Functions to apply in sequence.
# 
#     Returns
#     -------
#     callable
#         A single hook that applies all functions.
# 
#     Examples
#     --------
#     >>> from scitex.msword.utils import (
#     ...     link_captions_to_images,
#     ...     normalize_section_headings,
#     ...     create_post_import_hook,
#     ... )
#     >>> hook = create_post_import_hook(
#     ...     link_captions_to_images,
#     ...     normalize_section_headings,
#     ... )
#     >>> # Use with custom profile
#     >>> profile.post_import_hooks = [hook]
#     """
#     def composite_hook(doc: Dict[str, Any]) -> Dict[str, Any]:
#         for func in functions:
#             doc = func(doc)
#         return doc
#     return composite_hook
# 
# 
# __all__ = [
#     "link_captions_to_images",
#     "link_captions_to_images_by_proximity",
#     "normalize_section_headings",
#     "validate_document",
#     "create_post_import_hook",
# ]

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/msword/utils.py
# --------------------------------------------------------------------------------
