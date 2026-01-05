#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.results._SectionReadResponse."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.results._SectionReadResponse import SectionReadResponse


class TestSectionReadResponseCreation:
    """Tests for SectionReadResponse instantiation."""

    def test_required_fields(self):
        """Verify required fields are set correctly."""
        response = SectionReadResponse(
            success=True,
            content="\\section{Introduction}",
            section_name="introduction",
            section_id="manuscript/introduction",
            doc_type="manuscript",
        )
        assert response.success is True
        assert response.content == "\\section{Introduction}"
        assert response.section_name == "introduction"
        assert response.section_id == "manuscript/introduction"
        assert response.doc_type == "manuscript"

    def test_optional_fields_defaults(self):
        """Verify optional fields have proper defaults."""
        response = SectionReadResponse(
            success=True,
            content="Content",
            section_name="abstract",
            section_id="manuscript/abstract",
            doc_type="manuscript",
        )
        assert response.file_path is None
        assert response.error is None

    def test_file_path_can_be_set(self):
        """Verify file_path can be explicitly set."""
        file_path = Path("/tmp/project/01_manuscript/contents/abstract.tex")
        response = SectionReadResponse(
            success=True,
            content="Content",
            section_name="abstract",
            section_id="manuscript/abstract",
            doc_type="manuscript",
            file_path=file_path,
        )
        assert response.file_path == file_path


class TestSectionReadResponseFactoryMethods:
    """Tests for SectionReadResponse factory methods."""

    def test_create_success(self):
        """Verify create_success factory method."""
        response = SectionReadResponse.create_success(
            content="LaTeX content",
            section_name="methods",
            section_id="manuscript/methods",
            doc_type="manuscript",
        )
        assert response.success is True
        assert response.content == "LaTeX content"
        assert response.error is None

    def test_create_success_with_file_path(self):
        """Verify create_success with file path."""
        file_path = Path("/tmp/methods.tex")
        response = SectionReadResponse.create_success(
            content="Content",
            section_name="methods",
            section_id="manuscript/methods",
            doc_type="manuscript",
            file_path=file_path,
        )
        assert response.file_path == file_path

    def test_create_failure(self):
        """Verify create_failure factory method."""
        response = SectionReadResponse.create_failure(
            section_id="manuscript/abstract",
            error_message="File not found",
        )
        assert response.success is False
        assert response.content == ""
        assert response.error == "File not found"
        assert response.doc_type == "manuscript"
        assert response.section_name == "abstract"

    def test_create_failure_parses_section_id(self):
        """Verify create_failure parses section_id correctly."""
        response = SectionReadResponse.create_failure(
            section_id="supplementary/results",
            error_message="Error",
        )
        assert response.doc_type == "supplementary"
        assert response.section_name == "results"

    def test_create_failure_simple_section_id(self):
        """Verify create_failure handles simple section_id."""
        response = SectionReadResponse.create_failure(
            section_id="abstract",
            error_message="Error",
        )
        assert response.doc_type == "manuscript"
        assert response.section_name == "abstract"


class TestSectionReadResponseToDict:
    """Tests for SectionReadResponse to_dict method."""

    def test_to_dict_contains_all_fields(self):
        """Verify to_dict includes all fields."""
        response = SectionReadResponse(
            success=True,
            content="Content",
            section_name="abstract",
            section_id="manuscript/abstract",
            doc_type="manuscript",
            file_path=Path("/tmp/abstract.tex"),
            error=None,
        )
        result = response.to_dict()

        assert result["success"] is True
        assert result["content"] == "Content"
        assert result["section_name"] == "abstract"
        assert result["section_id"] == "manuscript/abstract"
        assert result["doc_type"] == "manuscript"
        assert result["file_path"] == "/tmp/abstract.tex"
        assert result["error"] is None

    def test_to_dict_none_file_path(self):
        """Verify to_dict handles None file_path."""
        response = SectionReadResponse(
            success=True,
            content="Content",
            section_name="abstract",
            section_id="manuscript/abstract",
            doc_type="manuscript",
        )
        result = response.to_dict()
        assert result["file_path"] is None


class TestSectionReadResponseStr:
    """Tests for SectionReadResponse __str__ method."""

    def test_str_success(self):
        """Verify string representation for success."""
        response = SectionReadResponse.create_success(
            content="X" * 100,
            section_name="abstract",
            section_id="manuscript/abstract",
            doc_type="manuscript",
        )
        str_result = str(response)
        assert "manuscript/abstract" in str_result
        assert "100 chars" in str_result

    def test_str_failure(self):
        """Verify string representation for failure."""
        response = SectionReadResponse.create_failure(
            section_id="manuscript/abstract",
            error_message="File not found",
        )
        str_result = str(response)
        assert "Failed to read" in str_result
        assert "File not found" in str_result


class TestSectionReadResponseValidation:
    """Tests for SectionReadResponse validate method."""

    def test_validate_success_empty_content_raises(self):
        """Verify validate raises for success with empty content."""
        response = SectionReadResponse(
            success=True,
            content="",
            section_name="abstract",
            section_id="manuscript/abstract",
            doc_type="manuscript",
        )
        with pytest.raises(ValueError, match="content is empty"):
            response.validate()

    def test_validate_success_empty_content_allowed_for_pdf(self):
        """Verify validate allows empty content for compiled_pdf."""
        response = SectionReadResponse(
            success=True,
            content="",
            section_name="compiled_pdf",
            section_id="manuscript/compiled_pdf",
            doc_type="manuscript",
        )
        response.validate()

    def test_validate_failure_without_error_raises(self):
        """Verify validate raises for failure without error message."""
        response = SectionReadResponse(
            success=False,
            content="",
            section_name="abstract",
            section_id="manuscript/abstract",
            doc_type="manuscript",
            error=None,
        )
        with pytest.raises(ValueError, match="no error message"):
            response.validate()

    def test_validate_empty_section_name_raises(self):
        """Verify validate raises for empty section_name."""
        response = SectionReadResponse(
            success=True,
            content="Content",
            section_name="",
            section_id="manuscript/abstract",
            doc_type="manuscript",
        )
        with pytest.raises(ValueError, match="section_name cannot be empty"):
            response.validate()

    def test_validate_empty_section_id_raises(self):
        """Verify validate raises for empty section_id."""
        response = SectionReadResponse(
            success=True,
            content="Content",
            section_name="abstract",
            section_id="",
            doc_type="manuscript",
        )
        with pytest.raises(ValueError, match="section_id cannot be empty"):
            response.validate()

    def test_validate_invalid_doc_type_raises(self):
        """Verify validate raises for invalid doc_type."""
        response = SectionReadResponse(
            success=True,
            content="Content",
            section_name="abstract",
            section_id="invalid/abstract",
            doc_type="invalid",
        )
        with pytest.raises(ValueError, match="Invalid doc_type"):
            response.validate()

    def test_validate_valid_doc_types(self):
        """Verify validate passes for all valid doc_types."""
        for doc_type in ["manuscript", "supplementary", "revision", "shared"]:
            response = SectionReadResponse(
                success=True,
                content="Content",
                section_name="abstract",
                section_id=f"{doc_type}/abstract",
                doc_type=doc_type,
            )
            response.validate()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
