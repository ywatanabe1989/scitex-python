#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: tests/scitex/scholar/storage/test__BibTeXValidator.py

"""Tests for BibTeXValidator module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# =============================================================================
# Test ValidationSeverity Enum
# =============================================================================
class TestValidationSeverity:
    """Tests for ValidationSeverity enum."""

    def test_severity_values(self):
        """Test that severity enum has correct values."""
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.INFO.value == "info"

    def test_all_severities_defined(self):
        """Test that all expected severity levels are defined."""
        severities = list(ValidationSeverity)
        assert len(severities) == 3
        assert ValidationSeverity.ERROR in severities
        assert ValidationSeverity.WARNING in severities
        assert ValidationSeverity.INFO in severities


# =============================================================================
# Test ValidationIssue Dataclass
# =============================================================================
class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_issue_creation_minimal(self):
        """Test creating an issue with minimal fields."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Test error message",
        )
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Test error message"
        assert issue.line_number is None
        assert issue.entry_key is None
        assert issue.field_name is None

    def test_issue_creation_full(self):
        """Test creating an issue with all fields."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Missing field",
            line_number=42,
            entry_key="smith2020",
            field_name="year",
        )
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.message == "Missing field"
        assert issue.line_number == 42
        assert issue.entry_key == "smith2020"
        assert issue.field_name == "year"

    def test_issue_str_minimal(self):
        """Test string representation with minimal fields."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="File not found",
        )
        result = str(issue)
        assert "[ERROR]" in result
        assert "File not found" in result

    def test_issue_str_with_line_number(self):
        """Test string representation with line number."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Unbalanced braces",
            line_number=15,
        )
        result = str(issue)
        assert "[WARNING]" in result
        assert "Line 15:" in result
        assert "Unbalanced braces" in result

    def test_issue_str_with_entry_key(self):
        """Test string representation with entry key."""
        issue = ValidationIssue(
            severity=ValidationSeverity.INFO,
            message="Non-standard year",
            entry_key="jones2021",
        )
        result = str(issue)
        assert "[INFO]" in result
        assert "Entry 'jones2021':" in result

    def test_issue_str_with_field_name(self):
        """Test string representation with field name."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Empty value",
            field_name="author",
        )
        result = str(issue)
        assert "Field 'author':" in result

    def test_issue_str_full(self):
        """Test string representation with all fields."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Duplicate key",
            line_number=100,
            entry_key="duplicate",
            field_name="title",
        )
        result = str(issue)
        assert "[ERROR]" in result
        assert "Line 100:" in result
        assert "Entry 'duplicate':" in result
        assert "Field 'title':" in result
        assert "Duplicate key" in result


# =============================================================================
# Test ValidationResult Dataclass
# =============================================================================
class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_creation_valid(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.file_path is None
        assert result.issues == []
        assert result.entry_count == 0
        assert result.duplicate_keys == []

    def test_result_creation_with_file_path(self):
        """Test creating result with file path."""
        result = ValidationResult(is_valid=True, file_path="/path/to/file.bib")
        assert result.file_path == "/path/to/file.bib"

    def test_result_errors_property(self):
        """Test errors property filters correctly."""
        issues = [
            ValidationIssue(severity=ValidationSeverity.ERROR, message="Error 1"),
            ValidationIssue(severity=ValidationSeverity.WARNING, message="Warning 1"),
            ValidationIssue(severity=ValidationSeverity.ERROR, message="Error 2"),
            ValidationIssue(severity=ValidationSeverity.INFO, message="Info 1"),
        ]
        result = ValidationResult(is_valid=False, issues=issues)
        errors = result.errors
        assert len(errors) == 2
        assert all(e.severity == ValidationSeverity.ERROR for e in errors)

    def test_result_warnings_property(self):
        """Test warnings property filters correctly."""
        issues = [
            ValidationIssue(severity=ValidationSeverity.ERROR, message="Error 1"),
            ValidationIssue(severity=ValidationSeverity.WARNING, message="Warning 1"),
            ValidationIssue(severity=ValidationSeverity.WARNING, message="Warning 2"),
            ValidationIssue(severity=ValidationSeverity.INFO, message="Info 1"),
        ]
        result = ValidationResult(is_valid=False, issues=issues)
        warnings = result.warnings
        assert len(warnings) == 2
        assert all(w.severity == ValidationSeverity.WARNING for w in warnings)

    def test_result_has_errors_true(self):
        """Test has_errors returns True when errors exist."""
        result = ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(severity=ValidationSeverity.ERROR, message="Error")
            ],
        )
        assert result.has_errors is True

    def test_result_has_errors_false(self):
        """Test has_errors returns False when no errors."""
        result = ValidationResult(
            is_valid=True,
            issues=[
                ValidationIssue(severity=ValidationSeverity.WARNING, message="Warning")
            ],
        )
        assert result.has_errors is False

    def test_result_has_warnings_true(self):
        """Test has_warnings returns True when warnings exist."""
        result = ValidationResult(
            is_valid=True,
            issues=[
                ValidationIssue(severity=ValidationSeverity.WARNING, message="Warning")
            ],
        )
        assert result.has_warnings is True

    def test_result_has_warnings_false(self):
        """Test has_warnings returns False when no warnings."""
        result = ValidationResult(is_valid=True, issues=[])
        assert result.has_warnings is False

    def test_result_str_valid(self):
        """Test string representation for valid result."""
        result = ValidationResult(is_valid=True, entry_count=5)
        result_str = str(result)
        assert "VALID" in result_str
        assert "5 entries" in result_str

    def test_result_str_invalid(self):
        """Test string representation for invalid result."""
        result = ValidationResult(is_valid=False, entry_count=3)
        result_str = str(result)
        assert "INVALID" in result_str

    def test_result_str_with_file_path(self):
        """Test string representation includes file path."""
        result = ValidationResult(is_valid=True, file_path="test.bib", entry_count=2)
        result_str = str(result)
        assert "test.bib:" in result_str

    def test_result_str_with_duplicate_keys(self):
        """Test string representation includes duplicate keys."""
        result = ValidationResult(
            is_valid=False, entry_count=3, duplicate_keys=["key1", "key2"]
        )
        result_str = str(result)
        assert "Duplicate keys: key1, key2" in result_str

    def test_result_str_with_issues(self):
        """Test string representation includes issues."""
        result = ValidationResult(
            is_valid=False,
            entry_count=1,
            issues=[
                ValidationIssue(severity=ValidationSeverity.ERROR, message="Test error")
            ],
        )
        result_str = str(result)
        assert "[ERROR]" in result_str
        assert "Test error" in result_str


# =============================================================================
# Test Constants
# =============================================================================
class TestConstants:
    """Tests for module constants."""

    def test_required_fields_article(self):
        """Test required fields for article type."""
        assert "author" in REQUIRED_FIELDS["article"]
        assert "title" in REQUIRED_FIELDS["article"]
        assert "journal" in REQUIRED_FIELDS["article"]
        assert "year" in REQUIRED_FIELDS["article"]

    def test_required_fields_book(self):
        """Test required fields for book type."""
        assert "author" in REQUIRED_FIELDS["book"]
        assert "title" in REQUIRED_FIELDS["book"]
        assert "publisher" in REQUIRED_FIELDS["book"]
        assert "year" in REQUIRED_FIELDS["book"]

    def test_required_fields_inproceedings(self):
        """Test required fields for inproceedings type."""
        assert "author" in REQUIRED_FIELDS["inproceedings"]
        assert "title" in REQUIRED_FIELDS["inproceedings"]
        assert "booktitle" in REQUIRED_FIELDS["inproceedings"]
        assert "year" in REQUIRED_FIELDS["inproceedings"]

    def test_required_fields_misc_empty(self):
        """Test that misc type has no required fields."""
        assert REQUIRED_FIELDS["misc"] == []

    def test_valid_entry_types_includes_common_types(self):
        """Test that valid entry types includes common types."""
        assert "article" in VALID_ENTRY_TYPES
        assert "book" in VALID_ENTRY_TYPES
        assert "inproceedings" in VALID_ENTRY_TYPES
        assert "misc" in VALID_ENTRY_TYPES

    def test_valid_entry_types_includes_extended_types(self):
        """Test that valid entry types includes extended types."""
        assert "online" in VALID_ENTRY_TYPES
        assert "software" in VALID_ENTRY_TYPES
        assert "dataset" in VALID_ENTRY_TYPES


# =============================================================================
# Test BibTeXValidator Initialization
# =============================================================================
class TestBibTeXValidatorInit:
    """Tests for BibTeXValidator initialization."""

    def test_init_default(self):
        """Test initialization with default settings."""
        validator = BibTeXValidator()
        assert validator.strict is False

    def test_init_strict(self):
        """Test initialization with strict mode."""
        validator = BibTeXValidator(strict=True)
        assert validator.strict is True


# =============================================================================
# Test BibTeXValidator.validate_file
# =============================================================================
class TestBibTeXValidatorValidateFile:
    """Tests for BibTeXValidator.validate_file method."""

    def test_validate_file_not_exists(self, tmp_path):
        """Test validation of non-existent file."""
        validator = BibTeXValidator()
        result = validator.validate_file(tmp_path / "nonexistent.bib")
        assert result.is_valid is False
        assert result.has_errors is True
        assert "does not exist" in result.errors[0].message

    def test_validate_file_wrong_extension(self, tmp_path):
        """Test validation of file with wrong extension."""
        test_file = tmp_path / "test.txt"
        test_file.write_text(
            "@article{test, author={Author}, title={Title}, year={2020}}"
        )
        validator = BibTeXValidator()
        result = validator.validate_file(test_file)
        assert result.has_warnings is True
        assert any("extension" in w.message for w in result.warnings)

    def test_validate_file_valid_bibtex(self, tmp_path):
        """Test validation of valid BibTeX file."""
        test_file = tmp_path / "test.bib"
        test_file.write_text(
            """@article{smith2020,
    author = {Smith, John},
    title = {A Great Paper},
    journal = {Journal of Testing},
    year = {2020}
}"""
        )
        validator = BibTeXValidator()
        result = validator.validate_file(test_file)
        assert result.is_valid is True
        assert result.entry_count == 1

    def test_validate_file_utf8_encoding(self, tmp_path):
        """Test validation of UTF-8 encoded file."""
        test_file = tmp_path / "test.bib"
        test_file.write_text(
            "@article{test, author={Müller}, title={Über alles}, year={2020}}",
            encoding="utf-8",
        )
        validator = BibTeXValidator()
        result = validator.validate_file(test_file)
        assert result.is_valid is True

    def test_validate_file_latin1_encoding(self, tmp_path):
        """Test validation of latin-1 encoded file."""
        test_file = tmp_path / "test.bib"
        # Write bytes that are valid latin-1 but invalid UTF-8
        test_file.write_bytes(
            b"@article{test, author={M\xfcller}, title={Test}, year={2020}}"
        )
        validator = BibTeXValidator()
        result = validator.validate_file(test_file)
        assert result.has_warnings is True
        assert any("latin-1" in w.message for w in result.warnings)

    def test_validate_file_sets_file_path(self, tmp_path):
        """Test that file path is set in result."""
        test_file = tmp_path / "test.bib"
        test_file.write_text("@article{test, author={A}, title={T}, year={2020}}")
        validator = BibTeXValidator()
        result = validator.validate_file(test_file)
        assert result.file_path == str(test_file)


# =============================================================================
# Test BibTeXValidator.validate_content
# =============================================================================
class TestBibTeXValidatorValidateContent:
    """Tests for BibTeXValidator.validate_content method."""

    def test_validate_content_empty(self):
        """Test validation of empty content."""
        validator = BibTeXValidator()
        result = validator.validate_content("")
        assert result.has_warnings is True
        assert any("empty" in w.message for w in result.warnings)

    def test_validate_content_whitespace_only(self):
        """Test validation of whitespace-only content."""
        validator = BibTeXValidator()
        result = validator.validate_content("   \n\t  ")
        assert result.has_warnings is True
        assert any("empty" in w.message for w in result.warnings)

    def test_validate_content_single_article(self):
        """Test validation of single article entry."""
        content = """@article{test2020,
    author = {Test Author},
    title = {Test Title},
    journal = {Test Journal},
    year = {2020}
}"""
        validator = BibTeXValidator()
        result = validator.validate_content(content)
        assert result.is_valid is True
        assert result.entry_count == 1

    def test_validate_content_multiple_entries(self):
        """Test validation of multiple entries."""
        content = """
@article{entry1, author={A1}, title={T1}, journal={J1}, year={2020}}
@book{entry2, author={A2}, title={T2}, publisher={P2}, year={2021}}
@misc{entry3, title={T3}}
"""
        validator = BibTeXValidator()
        result = validator.validate_content(content)
        assert result.entry_count == 3

    def test_validate_content_with_existing_result(self):
        """Test validation with existing result object."""
        existing = ValidationResult(
            is_valid=True,
            file_path="/test/path.bib",
            issues=[
                ValidationIssue(severity=ValidationSeverity.INFO, message="Existing")
            ],
        )
        validator = BibTeXValidator()
        result = validator.validate_content(
            "@article{test, author={A}, title={T}, year={2020}}", result=existing
        )
        assert result.file_path == "/test/path.bib"
        assert len(result.issues) >= 1


# =============================================================================
# Test BibTeXValidator._check_brace_balance
# =============================================================================
class TestBibTeXValidatorBraceBalance:
    """Tests for brace balance checking."""

    def test_balanced_braces(self):
        """Test that balanced braces produce no issues."""
        validator = BibTeXValidator()
        content = "@article{test, author={Author Name}, title={Title}}"
        issues = validator._check_brace_balance(content)
        assert len(issues) == 0

    def test_unclosed_brace(self):
        """Test detection of unclosed brace."""
        validator = BibTeXValidator()
        content = "@article{test, author={Author Name, title={Title}}"
        issues = validator._check_brace_balance(content)
        assert len(issues) > 0
        assert any("Unclosed" in i.message for i in issues)

    def test_unexpected_closing_brace(self):
        """Test detection of unexpected closing brace."""
        validator = BibTeXValidator()
        content = "@article{test, author=Author Name}, title={Title}}"
        issues = validator._check_brace_balance(content)
        assert len(issues) > 0
        assert any("Unexpected" in i.message for i in issues)

    def test_nested_braces(self):
        """Test that nested braces are handled correctly."""
        validator = BibTeXValidator()
        content = "@article{test, title={Title with {nested {braces}}}}"
        issues = validator._check_brace_balance(content)
        assert len(issues) == 0

    def test_braces_in_string_ignored(self):
        """Test that braces inside quoted strings are handled."""
        validator = BibTeXValidator()
        content = '@article{test, author="Author with { brace"}'
        issues = validator._check_brace_balance(content)
        # The content has valid outer braces
        assert len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0


# =============================================================================
# Test BibTeXValidator._parse_entries
# =============================================================================
class TestBibTeXValidatorParseEntries:
    """Tests for entry parsing."""

    def test_parse_article_entry(self):
        """Test parsing an article entry."""
        validator = BibTeXValidator()
        content = "@article{smith2020, author={Smith}, title={Title}, year={2020}}"
        entries, issues = validator._parse_entries(content)
        assert len(entries) == 1
        assert entries[0]["type"] == "article"
        assert entries[0]["key"] == "smith2020"

    def test_parse_entry_fields(self):
        """Test that entry fields are parsed correctly."""
        validator = BibTeXValidator()
        content = "@book{jones2021, author={Jones}, title={Book Title}, publisher={Pub}, year={2021}}"
        entries, issues = validator._parse_entries(content)
        fields = entries[0]["fields"]
        assert fields.get("author") == "Jones"
        assert fields.get("title") == "Book Title"
        assert fields.get("publisher") == "Pub"
        assert fields.get("year") == "2021"

    def test_parse_entry_no_key(self):
        """Test detection of entry without key."""
        validator = BibTeXValidator()
        content = "@article{, author={A}, title={T}, year={2020}}"
        entries, issues = validator._parse_entries(content)
        assert any(i.severity == ValidationSeverity.ERROR for i in issues)
        assert any("no key" in i.message for i in issues)

    def test_parse_unknown_entry_type(self):
        """Test detection of unknown entry type."""
        validator = BibTeXValidator()
        content = "@unknowntype{key, author={A}, title={T}}"
        entries, issues = validator._parse_entries(content)
        assert any("Unknown entry type" in i.message for i in issues)

    def test_parse_special_entry_types(self):
        """Test that special entry types are accepted."""
        validator = BibTeXValidator()
        content = """
@string{journal = "Test Journal"}
@comment{This is a comment}
@preamble{"Some preamble text"}
"""
        entries, issues = validator._parse_entries(content)
        # These should not produce unknown type warnings
        assert not any("Unknown entry type" in i.message for i in issues)

    def test_parse_entry_unusual_key_characters(self):
        """Test warning for entry keys with unusual characters."""
        validator = BibTeXValidator()
        # Use characters that are captured but unusual (not spaces, as those terminate the key capture)
        content = "@article{key#special@char, author={A}, title={T}, year={2020}}"
        entries, issues = validator._parse_entries(content)
        assert any("unusual characters" in i.message for i in issues)

    def test_parse_entry_line_number(self):
        """Test that line numbers are tracked correctly."""
        validator = BibTeXValidator()
        content = """
line1
line2
@article{test, author={A}, title={T}, year={2020}}
"""
        entries, issues = validator._parse_entries(content)
        assert entries[0]["line_number"] == 4


# =============================================================================
# Test BibTeXValidator._extract_entry_content
# =============================================================================
class TestBibTeXValidatorExtractContent:
    """Tests for entry content extraction."""

    def test_extract_simple_content(self):
        """Test extracting simple entry content."""
        validator = BibTeXValidator()
        content = "author={Author}, title={Title}}"
        result = validator._extract_entry_content(content, 0)
        assert result is not None
        assert "author={Author}" in result

    def test_extract_nested_braces(self):
        """Test extracting content with nested braces."""
        validator = BibTeXValidator()
        content = "title={Title with {nested} braces}}"
        result = validator._extract_entry_content(content, 0)
        assert result is not None
        assert "nested" in result

    def test_extract_unclosed_returns_none(self):
        """Test that unclosed braces return None."""
        validator = BibTeXValidator()
        content = "author={Author, title={Title}"
        result = validator._extract_entry_content(content, 0)
        assert result is None


# =============================================================================
# Test BibTeXValidator._parse_fields
# =============================================================================
class TestBibTeXValidatorParseFields:
    """Tests for field parsing."""

    def test_parse_braced_value(self):
        """Test parsing values in braces."""
        validator = BibTeXValidator()
        content = "author = {John Smith}"
        fields = validator._parse_fields(content)
        assert fields.get("author") == "John Smith"

    def test_parse_quoted_value(self):
        """Test parsing values in quotes."""
        validator = BibTeXValidator()
        content = 'title = "A Great Paper"'
        fields = validator._parse_fields(content)
        assert fields.get("title") == "A Great Paper"

    def test_parse_numeric_value(self):
        """Test parsing numeric values."""
        validator = BibTeXValidator()
        content = "year = 2020"
        fields = validator._parse_fields(content)
        assert fields.get("year") == "2020"

    def test_parse_multiple_fields(self):
        """Test parsing multiple fields."""
        validator = BibTeXValidator()
        content = "author = {Smith}, title = {Title}, year = 2021"
        fields = validator._parse_fields(content)
        assert fields.get("author") == "Smith"
        assert fields.get("title") == "Title"
        assert fields.get("year") == "2021"

    def test_parse_field_case_insensitive(self):
        """Test that field names are lowercased."""
        validator = BibTeXValidator()
        content = "AUTHOR = {Smith}, Title = {Test}"
        fields = validator._parse_fields(content)
        assert "author" in fields
        assert "title" in fields


# =============================================================================
# Test BibTeXValidator._validate_entry
# =============================================================================
class TestBibTeXValidatorValidateEntry:
    """Tests for individual entry validation."""

    def test_validate_complete_article(self):
        """Test validation of complete article entry."""
        validator = BibTeXValidator()
        entry = {
            "type": "article",
            "key": "test2020",
            "fields": {
                "author": "Smith, John",
                "title": "Test Paper",
                "journal": "Test Journal",
                "year": "2020",
            },
            "line_number": 1,
        }
        issues = validator._validate_entry(entry)
        # No missing required field issues
        assert not any("Missing recommended field" in i.message for i in issues)

    def test_validate_missing_required_fields(self):
        """Test detection of missing required fields."""
        validator = BibTeXValidator()
        entry = {
            "type": "article",
            "key": "test2020",
            "fields": {"title": "Test Paper"},
            "line_number": 1,
        }
        issues = validator._validate_entry(entry)
        assert any("Missing recommended field: author" in i.message for i in issues)
        assert any("Missing recommended field: journal" in i.message for i in issues)
        assert any("Missing recommended field: year" in i.message for i in issues)

    def test_validate_editor_substitutes_author(self):
        """Test that editor can substitute for author."""
        validator = BibTeXValidator()
        entry = {
            "type": "book",
            "key": "test2020",
            "fields": {
                "editor": "Smith, John",
                "title": "Test Book",
                "publisher": "Publisher",
                "year": "2020",
            },
            "line_number": 1,
        }
        issues = validator._validate_entry(entry)
        assert not any("Missing recommended field: author" in i.message for i in issues)

    def test_validate_empty_required_field(self):
        """Test detection of empty required fields."""
        validator = BibTeXValidator()
        entry = {
            "type": "article",
            "key": "test2020",
            "fields": {"author": "", "title": "Test", "journal": "J", "year": "2020"},
            "line_number": 1,
        }
        issues = validator._validate_entry(entry)
        assert any("Empty value for field: author" in i.message for i in issues)

    def test_validate_standard_year_format(self):
        """Test that standard year format produces no issues."""
        validator = BibTeXValidator()
        entry = {
            "type": "misc",
            "key": "test",
            "fields": {"year": "2020"},
            "line_number": 1,
        }
        issues = validator._validate_entry(entry)
        assert not any("year" in i.message.lower() for i in issues)

    def test_validate_nonstandard_year_format(self):
        """Test detection of non-standard year format."""
        validator = BibTeXValidator()
        entry = {
            "type": "misc",
            "key": "test",
            "fields": {"year": "Twenty Twenty"},
            "line_number": 1,
        }
        issues = validator._validate_entry(entry)
        assert any("Non-standard year format" in i.message for i in issues)

    def test_validate_year_range_accepted(self):
        """Test that year ranges are accepted."""
        validator = BibTeXValidator()
        entry = {
            "type": "misc",
            "key": "test",
            "fields": {"year": "2020-2021"},
            "line_number": 1,
        }
        issues = validator._validate_entry(entry)
        assert not any("Non-standard year" in i.message for i in issues)

    def test_validate_valid_doi(self):
        """Test validation of valid DOI."""
        validator = BibTeXValidator()
        entry = {
            "type": "misc",
            "key": "test",
            "fields": {"doi": "10.1234/abc123"},
            "line_number": 1,
        }
        issues = validator._validate_entry(entry)
        assert not any("Invalid DOI" in i.message for i in issues)

    def test_validate_invalid_doi(self):
        """Test detection of invalid DOI format."""
        validator = BibTeXValidator()
        entry = {
            "type": "misc",
            "key": "test",
            "fields": {"doi": "invalid-doi"},
            "line_number": 1,
        }
        issues = validator._validate_entry(entry)
        assert any("Invalid DOI format" in i.message for i in issues)

    def test_validate_doi_with_url(self):
        """Test that DOI with doi.org URL is accepted."""
        validator = BibTeXValidator()
        entry = {
            "type": "misc",
            "key": "test",
            "fields": {"doi": "https://doi.org/10.1234/abc"},
            "line_number": 1,
        }
        issues = validator._validate_entry(entry)
        assert not any("Invalid DOI" in i.message for i in issues)

    def test_validate_special_entries_skipped(self):
        """Test that special entries are skipped."""
        validator = BibTeXValidator()
        for entry_type in ["string", "comment", "preamble"]:
            entry = {"type": entry_type, "key": "", "fields": {}, "line_number": 1}
            issues = validator._validate_entry(entry)
            assert len(issues) == 0


# =============================================================================
# Test Duplicate Key Detection
# =============================================================================
class TestBibTeXValidatorDuplicates:
    """Tests for duplicate key detection."""

    def test_no_duplicates(self):
        """Test content with no duplicate keys."""
        validator = BibTeXValidator()
        content = """
@article{entry1, author={A1}, title={T1}, year={2020}}
@article{entry2, author={A2}, title={T2}, year={2021}}
"""
        result = validator.validate_content(content)
        assert len(result.duplicate_keys) == 0

    def test_detect_duplicate_keys(self):
        """Test detection of duplicate keys."""
        validator = BibTeXValidator()
        content = """
@article{duplicate, author={A1}, title={T1}, year={2020}}
@book{duplicate, author={A2}, title={T2}, year={2021}}
"""
        result = validator.validate_content(content)
        assert "duplicate" in result.duplicate_keys
        assert result.has_errors is True

    def test_duplicate_detection_case_insensitive(self):
        """Test that duplicate detection is case-insensitive."""
        validator = BibTeXValidator()
        content = """
@article{Smith2020, author={A1}, title={T1}, year={2020}}
@book{smith2020, author={A2}, title={T2}, year={2021}}
"""
        result = validator.validate_content(content)
        assert len(result.duplicate_keys) > 0


# =============================================================================
# Test Strict Mode
# =============================================================================
class TestBibTeXValidatorStrictMode:
    """Tests for strict validation mode."""

    def test_non_strict_warnings_dont_invalidate(self):
        """Test that warnings don't invalidate in non-strict mode."""
        validator = BibTeXValidator(strict=False)
        content = "@article{test, title={Only Title}}"  # Missing required fields
        result = validator.validate_content(content)
        # Has warnings but still valid
        assert result.has_warnings is True
        assert result.is_valid is True  # Non-strict ignores warnings

    def test_strict_warnings_invalidate(self):
        """Test that warnings invalidate in strict mode."""
        validator = BibTeXValidator(strict=True)
        content = "@article{test, title={Only Title}}"  # Missing required fields
        result = validator.validate_content(content)
        assert result.has_warnings is True
        assert result.is_valid is False  # Strict treats warnings as errors


# =============================================================================
# Test validate_files
# =============================================================================
class TestBibTeXValidatorValidateFiles:
    """Tests for validating multiple files."""

    def test_validate_multiple_files(self, tmp_path):
        """Test validating multiple files."""
        file1 = tmp_path / "file1.bib"
        file2 = tmp_path / "file2.bib"
        file1.write_text("@article{entry1, author={A}, title={T}, year={2020}}")
        file2.write_text(
            "@book{entry2, author={A}, title={T}, publisher={P}, year={2021}}"
        )

        validator = BibTeXValidator()
        results = validator.validate_files([file1, file2])
        assert len(results) == 2
        assert all(r.is_valid for r in results)

    def test_validate_files_with_errors(self, tmp_path):
        """Test validating files with errors."""
        valid_file = tmp_path / "valid.bib"
        invalid_file = tmp_path / "invalid.bib"
        valid_file.write_text("@article{test, author={A}, title={T}, year={2020}}")
        invalid_file.write_text("@article{test, author={A}, title={T}}")  # Unclosed

        validator = BibTeXValidator()
        results = validator.validate_files([valid_file, invalid_file])
        assert len(results) == 2


# =============================================================================
# Test validate_before_merge
# =============================================================================
class TestBibTeXValidatorValidateBeforeMerge:
    """Tests for pre-merge validation."""

    def test_merge_validation_valid(self, tmp_path):
        """Test pre-merge validation with valid files."""
        file1 = tmp_path / "file1.bib"
        file2 = tmp_path / "file2.bib"
        file1.write_text("@article{entry1, author={A}, title={T}, year={2020}}")
        file2.write_text(
            "@book{entry2, author={A}, title={T}, publisher={P}, year={2021}}"
        )

        validator = BibTeXValidator()
        can_merge, results = validator.validate_before_merge([file1, file2])
        assert can_merge is True
        assert len(results) == 2

    def test_merge_validation_cross_file_duplicates(self, tmp_path):
        """Test detection of cross-file duplicate keys."""
        file1 = tmp_path / "file1.bib"
        file2 = tmp_path / "file2.bib"
        file1.write_text("@article{samekey, author={A}, title={T}, year={2020}}")
        file2.write_text(
            "@book{samekey, author={A}, title={T}, publisher={P}, year={2021}}"
        )

        validator = BibTeXValidator()
        can_merge, results = validator.validate_before_merge([file1, file2])
        assert can_merge is False
        assert any("Cross-file duplicate" in str(r) for r in results)

    def test_merge_validation_invalid_file(self, tmp_path):
        """Test pre-merge validation with invalid file."""
        valid_file = tmp_path / "valid.bib"
        nonexistent = tmp_path / "nonexistent.bib"
        valid_file.write_text("@article{test, author={A}, title={T}, year={2020}}")

        validator = BibTeXValidator()
        can_merge, results = validator.validate_before_merge([valid_file, nonexistent])
        assert can_merge is False


# =============================================================================
# Test Convenience Functions
# =============================================================================
class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_validate_bibtex_file(self, tmp_path):
        """Test validate_bibtex_file convenience function."""
        test_file = tmp_path / "test.bib"
        test_file.write_text("@article{test, author={A}, title={T}, year={2020}}")
        result = validate_bibtex_file(test_file)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_bibtex_file_strict(self, tmp_path):
        """Test validate_bibtex_file with strict mode."""
        test_file = tmp_path / "test.bib"
        test_file.write_text("@article{test, title={Only Title}}")  # Missing fields
        result = validate_bibtex_file(test_file, strict=True)
        assert result.is_valid is False

    def test_validate_bibtex_content(self):
        """Test validate_bibtex_content convenience function."""
        content = "@article{test, author={A}, title={T}, year={2020}}"
        result = validate_bibtex_content(content)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_bibtex_content_strict(self):
        """Test validate_bibtex_content with strict mode."""
        content = "@article{test, title={Only Title}}"  # Missing fields
        result = validate_bibtex_content(content, strict=True)
        assert result.is_valid is False


# =============================================================================
# Test Edge Cases
# =============================================================================
class TestBibTeXValidatorEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_large_file(self, tmp_path):
        """Test validation of large file."""
        test_file = tmp_path / "large.bib"
        entries = []
        for i in range(100):
            entries.append(
                f"@article{{entry{i}, author={{Author {i}}}, title={{Title {i}}}, journal={{J}}, year={{2020}}}}"
            )
        test_file.write_text("\n".join(entries))
        validator = BibTeXValidator()
        result = validator.validate_file(test_file)
        assert result.entry_count == 100

    def test_special_characters_in_values(self):
        """Test handling of special characters in values."""
        validator = BibTeXValidator()
        content = r"@article{test, author={O'Brien}, title={Test & Demo}, year={2020}}"
        result = validator.validate_content(content)
        assert result.entry_count == 1

    def test_multiline_values(self):
        """Test handling of multiline values."""
        validator = BibTeXValidator()
        content = """@article{test,
    author = {John Smith and
              Jane Doe},
    title = {A Very Long Title That
             Spans Multiple Lines},
    journal = {Test Journal},
    year = {2020}
}"""
        result = validator.validate_content(content)
        assert result.entry_count == 1

    def test_comments_in_file(self):
        """Test handling of comments in file."""
        validator = BibTeXValidator()
        content = """
% This is a comment
@article{test, author={A}, title={T}, journal={J}, year={2020}}
% Another comment
"""
        result = validator.validate_content(content)
        assert result.entry_count == 1

    def test_mixed_entry_types(self):
        """Test file with mixed entry types."""
        validator = BibTeXValidator()
        content = """
@article{a1, author={A}, title={T}, journal={J}, year={2020}}
@book{b1, author={A}, title={T}, publisher={P}, year={2020}}
@inproceedings{c1, author={A}, title={T}, booktitle={B}, year={2020}}
@misc{m1, title={T}}
@online{o1, title={T}, url={http://example.com}}
"""
        result = validator.validate_content(content)
        assert result.entry_count == 5
        assert result.is_valid is True


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
