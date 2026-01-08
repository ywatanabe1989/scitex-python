#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: tests/scitex/scholar/storage/test_bibtex_validator.py

"""Tests for BibTeX file validation."""

import tempfile
from pathlib import Path

import pytest

from scitex.scholar.storage._BibTeXValidator import (
    REQUIRED_FIELDS,
    VALID_ENTRY_TYPES,
    BibTeXValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    validate_bibtex_content,
    validate_bibtex_file,
)


class TestValidationSeverity:
    """Tests for ValidationSeverity enum."""

    def test_severity_values(self):
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.INFO.value == "info"


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_basic_issue(self):
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Test error message",
        )
        assert issue.severity == ValidationSeverity.ERROR
        assert "ERROR" in str(issue)
        assert "Test error message" in str(issue)

    def test_issue_with_line_number(self):
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Warning message",
            line_number=42,
        )
        assert "Line 42" in str(issue)
        assert "WARNING" in str(issue)

    def test_issue_with_entry_key(self):
        issue = ValidationIssue(
            severity=ValidationSeverity.INFO,
            message="Info message",
            entry_key="smith2020",
        )
        assert "Entry 'smith2020'" in str(issue)

    def test_issue_with_field_name(self):
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Field warning",
            entry_key="test",
            field_name="author",
        )
        assert "Field 'author'" in str(issue)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        result = ValidationResult(is_valid=True, entry_count=5)
        assert result.is_valid
        assert result.entry_count == 5
        assert not result.has_errors
        assert not result.has_warnings
        assert "VALID" in str(result)

    def test_invalid_result_with_errors(self):
        result = ValidationResult(is_valid=False)
        result.issues.append(ValidationIssue(ValidationSeverity.ERROR, "Error 1"))
        result.issues.append(ValidationIssue(ValidationSeverity.ERROR, "Error 2"))
        assert not result.is_valid
        assert result.has_errors
        assert len(result.errors) == 2

    def test_result_with_warnings(self):
        result = ValidationResult(is_valid=True)
        result.issues.append(ValidationIssue(ValidationSeverity.WARNING, "Warning 1"))
        assert result.has_warnings
        assert len(result.warnings) == 1

    def test_result_with_file_path(self):
        result = ValidationResult(is_valid=True, file_path="/path/to/file.bib")
        assert "/path/to/file.bib" in str(result)

    def test_result_with_duplicate_keys(self):
        result = ValidationResult(is_valid=False)
        result.duplicate_keys = ["key1", "key2"]
        assert "key1" in str(result)
        assert "key2" in str(result)


class TestBibTeXValidator:
    """Tests for BibTeXValidator class."""

    @pytest.fixture
    def validator(self):
        return BibTeXValidator()

    @pytest.fixture
    def strict_validator(self):
        return BibTeXValidator(strict=True)

    # Basic validation tests
    def test_validate_empty_content(self, validator):
        result = validator.validate_content("")
        assert len(result.warnings) > 0
        assert any("empty" in str(w).lower() for w in result.warnings)

    def test_validate_whitespace_only(self, validator):
        result = validator.validate_content("   \n\t\n   ")
        assert len(result.warnings) > 0

    def test_validate_valid_article(self, validator):
        content = """
        @article{smith2020,
            author = {John Smith},
            title = {A Test Paper},
            journal = {Test Journal},
            year = {2020},
        }
        """
        result = validator.validate_content(content)
        assert result.is_valid
        assert result.entry_count == 1

    def test_validate_multiple_entries(self, validator):
        content = """
        @article{smith2020,
            author = {John Smith},
            title = {Paper One},
            journal = {Journal One},
            year = {2020},
        }
        @book{jones2021,
            author = {Jane Jones},
            title = {Book Title},
            publisher = {Publisher Name},
            year = {2021},
        }
        """
        result = validator.validate_content(content)
        assert result.is_valid
        assert result.entry_count == 2

    # Brace balance tests
    def test_unbalanced_braces_extra_open(self, validator):
        content = """
        @article{smith2020,
            author = {John Smith,
            title = {A Test Paper},
            year = {2020},
        }
        """
        result = validator.validate_content(content)
        assert result.has_errors
        assert any("brace" in str(e).lower() for e in result.errors)

    def test_unbalanced_braces_extra_close(self, validator):
        content = """
        @article{smith2020,
            author = {John Smith}},
            title = {A Test Paper},
            year = {2020},
        }
        """
        result = validator.validate_content(content)
        assert result.has_errors

    def test_balanced_nested_braces(self, validator):
        content = """
        @article{smith2020,
            author = {John Smith},
            title = {A {Nested} Title},
            journal = {Test Journal},
            year = {2020},
        }
        """
        result = validator.validate_content(content)
        assert result.is_valid

    # Entry type validation
    def test_unknown_entry_type(self, validator):
        content = """
        @unknown_type{test2020,
            title = {Test},
        }
        """
        result = validator.validate_content(content)
        assert any("unknown entry type" in str(w).lower() for w in result.warnings)

    def test_valid_entry_types(self, validator):
        for entry_type in ["article", "book", "inproceedings", "misc"]:
            content = f"""
            @{entry_type}{{test2020,
                title = {{Test}},
            }}
            """
            result = validator.validate_content(content)
            assert not any(
                "unknown entry type" in str(w).lower() for w in result.warnings
            )

    # Entry key validation
    def test_missing_entry_key(self, validator):
        content = """
        @article{,
            author = {John Smith},
            title = {Test},
        }
        """
        result = validator.validate_content(content)
        assert result.has_errors
        assert any("no key" in str(e).lower() for e in result.errors)

    def test_unusual_characters_in_key(self, validator):
        content = """
        @article{test@key!,
            title = {Test},
        }
        """
        result = validator.validate_content(content)
        assert any("unusual characters" in str(w).lower() for w in result.warnings)

    def test_valid_key_with_special_chars(self, validator):
        content = """
        @article{smith-jones_2020:paper,
            title = {Test},
        }
        """
        result = validator.validate_content(content)
        assert not any("unusual characters" in str(w).lower() for w in result.warnings)

    # Duplicate key detection
    def test_duplicate_keys(self, validator):
        content = """
        @article{smith2020,
            title = {Paper One},
        }
        @book{smith2020,
            title = {Paper Two},
        }
        """
        result = validator.validate_content(content)
        assert not result.is_valid
        assert "smith2020" in result.duplicate_keys

    def test_case_insensitive_duplicate_keys(self, validator):
        content = """
        @article{Smith2020,
            title = {Paper One},
        }
        @book{smith2020,
            title = {Paper Two},
        }
        """
        result = validator.validate_content(content)
        assert "smith2020" in result.duplicate_keys

    # Required fields validation
    def test_missing_required_fields_article(self, validator):
        content = """
        @article{smith2020,
            title = {Test Paper},
        }
        """
        result = validator.validate_content(content)
        assert any("author" in str(w).lower() for w in result.warnings)
        assert any("journal" in str(w).lower() for w in result.warnings)
        assert any("year" in str(w).lower() for w in result.warnings)

    def test_editor_satisfies_author_requirement(self, validator):
        content = """
        @book{collection2020,
            editor = {John Smith},
            title = {Edited Collection},
            publisher = {Publisher},
            year = {2020},
        }
        """
        result = validator.validate_content(content)
        assert not any(
            "author" in str(w).lower() and "missing" in str(w).lower()
            for w in result.warnings
        )

    def test_empty_required_field(self, validator):
        content = """
        @article{smith2020,
            author = {},
            title = {Test Paper},
            journal = {Test Journal},
            year = {2020},
        }
        """
        result = validator.validate_content(content)
        assert any(
            "empty" in str(w).lower() and "author" in str(w).lower()
            for w in result.warnings
        )

    # Year format validation
    def test_valid_year(self, validator):
        content = """
        @article{smith2020,
            author = {John Smith},
            title = {Test},
            journal = {Journal},
            year = {2020},
        }
        """
        result = validator.validate_content(content)
        assert not any(
            "year" in str(i).lower() and "format" in str(i).lower()
            for i in result.issues
        )

    def test_year_range(self, validator):
        content = """
        @article{smith2020,
            title = {Test},
            year = {2020-2021},
        }
        """
        result = validator.validate_content(content)
        # Year ranges should be allowed
        assert not any("non-standard year" in str(i).lower() for i in result.issues)

    def test_invalid_year_format(self, validator):
        content = """
        @article{smith2020,
            title = {Test},
            year = {twenty twenty},
        }
        """
        result = validator.validate_content(content)
        assert any("year" in str(i).lower() for i in result.issues)

    # DOI validation
    def test_valid_doi(self, validator):
        content = """
        @article{smith2020,
            title = {Test},
            doi = {10.1038/s41586-021-03819-2},
        }
        """
        result = validator.validate_content(content)
        assert not any(
            "doi" in str(w).lower() and "invalid" in str(w).lower()
            for w in result.warnings
        )

    def test_invalid_doi(self, validator):
        content = """
        @article{smith2020,
            title = {Test},
            doi = {invalid_doi_format},
        }
        """
        result = validator.validate_content(content)
        assert any(
            "doi" in str(w).lower() and "invalid" in str(w).lower()
            for w in result.warnings
        )

    def test_doi_with_url(self, validator):
        content = """
        @article{smith2020,
            title = {Test},
            doi = {https://doi.org/10.1038/test},
        }
        """
        result = validator.validate_content(content)
        # DOI URLs should be valid
        assert not any(
            "doi" in str(w).lower() and "invalid" in str(w).lower()
            for w in result.warnings
        )

    # Strict mode tests
    def test_strict_mode_warnings_as_errors(self, strict_validator):
        content = """
        @article{smith2020,
            title = {Test Paper},
        }
        """
        result = strict_validator.validate_content(content)
        assert not result.is_valid  # Warnings treated as errors in strict mode

    def test_non_strict_mode_warnings_allowed(self, validator):
        content = """
        @article{smith2020,
            title = {Test Paper},
        }
        """
        result = validator.validate_content(content)
        assert result.is_valid  # Warnings don't affect validity in normal mode

    # Special entry types
    def test_string_entry(self, validator):
        content = """
        @string{myjournal = "Test Journal"}
        @article{smith2020,
            title = {Test},
            journal = myjournal,
        }
        """
        result = validator.validate_content(content)
        # String entries should not cause errors
        assert not result.has_errors

    def test_comment_entry(self, validator):
        content = """
        @comment{This is a comment}
        @article{smith2020,
            title = {Test},
        }
        """
        result = validator.validate_content(content)
        assert not result.has_errors

    # File validation tests
    def test_validate_nonexistent_file(self, validator):
        result = validator.validate_file("/nonexistent/path/file.bib")
        assert not result.is_valid
        assert result.has_errors
        assert any("not exist" in str(e).lower() for e in result.errors)

    def test_validate_wrong_extension(self, validator):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"@article{test, title={Test}}")
            temp_path = f.name

        try:
            result = validator.validate_file(temp_path)
            assert any(".bib" in str(w) for w in result.warnings)
        finally:
            Path(temp_path).unlink()

    def test_validate_file_success(self, validator):
        content = """@article{smith2020,
            author = {John Smith},
            title = {Test Paper},
            journal = {Test Journal},
            year = {2020},
        }"""

        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False, mode="w") as f:
            f.write(content)
            temp_path = f.name

        try:
            result = validator.validate_file(temp_path)
            assert result.is_valid
            assert result.entry_count == 1
        finally:
            Path(temp_path).unlink()

    # Multiple file validation
    def test_validate_multiple_files(self, validator):
        content1 = "@article{test1, title={Test 1}}"
        content2 = "@article{test2, title={Test 2}}"

        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False, mode="w") as f1:
            f1.write(content1)
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False, mode="w") as f2:
            f2.write(content2)
            path2 = f2.name

        try:
            results = validator.validate_files([path1, path2])
            assert len(results) == 2
            assert all(r.is_valid for r in results)
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    # Pre-merge validation
    def test_validate_before_merge_success(self, validator):
        content1 = "@article{test1, title={Test 1}}"
        content2 = "@article{test2, title={Test 2}}"

        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False, mode="w") as f1:
            f1.write(content1)
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False, mode="w") as f2:
            f2.write(content2)
            path2 = f2.name

        try:
            can_merge, results = validator.validate_before_merge([path1, path2])
            assert can_merge
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    def test_validate_before_merge_cross_file_duplicates(self, validator):
        content1 = "@article{duplicate_key, title={Test 1}}"
        content2 = "@article{duplicate_key, title={Test 2}}"

        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False, mode="w") as f1:
            f1.write(content1)
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False, mode="w") as f2:
            f2.write(content2)
            path2 = f2.name

        try:
            can_merge, results = validator.validate_before_merge([path1, path2])
            assert not can_merge
            assert any(
                "cross-file" in str(i).lower() for r in results for i in r.issues
            )
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    # Field parsing tests
    def test_parse_braced_value(self, validator):
        content = """
        @article{test,
            title = {This is a {Nested} Value},
        }
        """
        result = validator.validate_content(content)
        assert result.entry_count == 1

    def test_parse_quoted_value(self, validator):
        content = """
        @article{test,
            title = "This is a quoted value",
        }
        """
        result = validator.validate_content(content)
        assert result.entry_count == 1

    def test_parse_numeric_value(self, validator):
        content = """
        @article{test,
            year = 2020,
            volume = 15,
        }
        """
        result = validator.validate_content(content)
        assert result.entry_count == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_bibtex_file(self):
        content = "@article{test, title={Test}}"
        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False, mode="w") as f:
            f.write(content)
            path = f.name

        try:
            result = validate_bibtex_file(path)
            assert isinstance(result, ValidationResult)
            assert result.is_valid
        finally:
            Path(path).unlink()

    def test_validate_bibtex_file_strict(self):
        content = "@article{test, title={Test}}"  # Missing required fields
        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False, mode="w") as f:
            f.write(content)
            path = f.name

        try:
            result = validate_bibtex_file(path, strict=True)
            assert not result.is_valid  # Strict mode fails on warnings
        finally:
            Path(path).unlink()

    def test_validate_bibtex_content(self):
        content = (
            "@article{test, author={Smith}, title={Test}, journal={J}, year={2020}}"
        )
        result = validate_bibtex_content(content)
        assert isinstance(result, ValidationResult)
        assert result.is_valid

    def test_validate_bibtex_content_strict(self):
        content = "@article{test, title={Test}}"
        result = validate_bibtex_content(content, strict=True)
        assert not result.is_valid


class TestRequiredFieldsDefinition:
    """Tests for required fields definitions."""

    def test_article_required_fields(self):
        assert "author" in REQUIRED_FIELDS["article"]
        assert "title" in REQUIRED_FIELDS["article"]
        assert "journal" in REQUIRED_FIELDS["article"]
        assert "year" in REQUIRED_FIELDS["article"]

    def test_book_required_fields(self):
        assert "author" in REQUIRED_FIELDS["book"]
        assert "title" in REQUIRED_FIELDS["book"]
        assert "publisher" in REQUIRED_FIELDS["book"]
        assert "year" in REQUIRED_FIELDS["book"]

    def test_misc_no_required_fields(self):
        assert len(REQUIRED_FIELDS["misc"]) == 0

    def test_valid_entry_types_includes_standard(self):
        standard_types = ["article", "book", "inproceedings", "misc", "phdthesis"]
        for t in standard_types:
            assert t in VALID_ENTRY_TYPES

    def test_valid_entry_types_includes_modern(self):
        modern_types = ["online", "software", "dataset"]
        for t in modern_types:
            assert t in VALID_ENTRY_TYPES


# EOF
