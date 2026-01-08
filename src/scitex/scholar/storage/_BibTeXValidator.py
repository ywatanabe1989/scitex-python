#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/scholar/storage/_BibTeXValidator.py

"""BibTeX file validation for syntax, structure, and content integrity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set, Tuple

from scitex import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Prevents parsing/merging
    WARNING = "warning"  # May cause issues
    INFO = "info"  # Informational


@dataclass
class ValidationIssue:
    """A single validation issue found in a BibTeX file."""

    severity: ValidationSeverity
    message: str
    line_number: Optional[int] = None
    entry_key: Optional[str] = None
    field_name: Optional[str] = None

    def __str__(self):
        parts = [f"[{self.severity.value.upper()}]"]
        if self.line_number:
            parts.append(f"Line {self.line_number}:")
        if self.entry_key:
            parts.append(f"Entry '{self.entry_key}':")
        if self.field_name:
            parts.append(f"Field '{self.field_name}':")
        parts.append(self.message)
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of BibTeX validation."""

    is_valid: bool
    file_path: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    entry_count: int = 0
    duplicate_keys: List[str] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def __str__(self):
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"Validation: {status} ({self.entry_count} entries)"]
        if self.file_path:
            lines[0] = f"{self.file_path}: {lines[0]}"
        if self.duplicate_keys:
            lines.append(f"Duplicate keys: {', '.join(self.duplicate_keys)}")
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


# Required fields for common entry types
REQUIRED_FIELDS = {
    "article": ["author", "title", "journal", "year"],
    "book": ["author", "title", "publisher", "year"],
    "inproceedings": ["author", "title", "booktitle", "year"],
    "conference": ["author", "title", "booktitle", "year"],
    "incollection": ["author", "title", "booktitle", "publisher", "year"],
    "inbook": ["author", "title", "publisher", "year"],
    "phdthesis": ["author", "title", "school", "year"],
    "mastersthesis": ["author", "title", "school", "year"],
    "techreport": ["author", "title", "institution", "year"],
    "misc": [],  # No required fields
    "unpublished": ["author", "title"],
    "manual": ["title"],
    "proceedings": ["title", "year"],
    "booklet": ["title"],
}

# Valid entry types (lowercase)
VALID_ENTRY_TYPES = set(REQUIRED_FIELDS.keys()) | {
    "online",
    "www",
    "electronic",
    "patent",
    "standard",
    "dataset",
    "software",
}


class BibTeXValidator:
    """Validates BibTeX files for syntax and content integrity."""

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, warnings are treated as errors
        """
        self.strict = strict

    def validate_file(self, file_path: str | Path) -> ValidationResult:
        """
        Validate a BibTeX file.

        Args:
            file_path: Path to the .bib file

        Returns:
            ValidationResult with issues found
        """
        file_path = Path(file_path)
        result = ValidationResult(is_valid=True, file_path=str(file_path))

        if not file_path.exists():
            result.is_valid = False
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"File does not exist: {file_path}",
                )
            )
            return result

        if not file_path.suffix.lower() == ".bib":
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"File extension is not .bib: {file_path.suffix}",
                )
            )

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="File uses latin-1 encoding instead of UTF-8",
                    )
                )
            except Exception as e:
                result.is_valid = False
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Cannot read file: {e}",
                    )
                )
                return result

        return self.validate_content(content, result)

    def validate_content(
        self, content: str, result: Optional[ValidationResult] = None
    ) -> ValidationResult:
        """
        Validate BibTeX content string.

        Args:
            content: BibTeX content as string
            result: Existing result to append to

        Returns:
            ValidationResult with issues found
        """
        if result is None:
            result = ValidationResult(is_valid=True)

        # Check for empty content
        if not content.strip():
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="File is empty or contains only whitespace",
                )
            )
            return result

        # Check brace balance
        brace_issues = self._check_brace_balance(content)
        result.issues.extend(brace_issues)

        # Parse and validate entries
        entries, parse_issues = self._parse_entries(content)
        result.issues.extend(parse_issues)
        result.entry_count = len(entries)

        # Check for duplicate keys
        keys = [e["key"] for e in entries if e.get("key")]
        seen_keys: Set[str] = set()
        for key in keys:
            if key.lower() in seen_keys:
                result.duplicate_keys.append(key)
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Duplicate entry key: {key}",
                        entry_key=key,
                    )
                )
            seen_keys.add(key.lower())

        # Validate each entry
        for entry in entries:
            entry_issues = self._validate_entry(entry)
            result.issues.extend(entry_issues)

        # Determine overall validity
        if self.strict:
            result.is_valid = not result.has_errors and not result.has_warnings
        else:
            result.is_valid = not result.has_errors

        return result

    def _check_brace_balance(self, content: str) -> List[ValidationIssue]:
        """Check that braces are balanced in the content."""
        issues = []
        brace_count = 0
        line_number = 1

        in_string = False
        prev_char = ""

        for char in content:
            if char == "\n":
                line_number += 1

            # Track string literals (simplistic - doesn't handle all edge cases)
            if char == '"' and prev_char != "\\":
                in_string = not in_string

            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count < 0:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                message="Unexpected closing brace",
                                line_number=line_number,
                            )
                        )
                        brace_count = 0  # Reset to continue checking

            prev_char = char

        if brace_count > 0:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Unclosed braces: {brace_count} opening brace(s) without matching close",
                )
            )

        return issues

    def _parse_entries(
        self, content: str
    ) -> Tuple[List[dict], List[ValidationIssue]]:
        """Parse BibTeX entries from content."""
        entries = []
        issues = []

        # Pattern to match entry starts
        entry_pattern = re.compile(
            r"@(\w+)\s*\{\s*([^,\s]*)\s*,", re.IGNORECASE | re.MULTILINE
        )

        # Find all entry starts
        for match in entry_pattern.finditer(content):
            entry_type = match.group(1).lower()
            entry_key = match.group(2).strip()
            start_pos = match.start()

            # Find line number
            line_number = content[:start_pos].count("\n") + 1

            # Validate entry type
            if entry_type not in VALID_ENTRY_TYPES and entry_type not in [
                "string",
                "comment",
                "preamble",
            ]:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Unknown entry type: @{entry_type}",
                        line_number=line_number,
                        entry_key=entry_key,
                    )
                )

            # Validate entry key
            if not entry_key:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Entry has no key",
                        line_number=line_number,
                    )
                )
            elif not re.match(r"^[\w\-:._]+$", entry_key):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Entry key contains unusual characters: {entry_key}",
                        line_number=line_number,
                        entry_key=entry_key,
                    )
                )

            # Extract entry content (find matching closing brace)
            entry_content = self._extract_entry_content(content, match.end())

            # Parse fields
            fields = self._parse_fields(entry_content) if entry_content else {}

            entries.append(
                {
                    "type": entry_type,
                    "key": entry_key,
                    "fields": fields,
                    "line_number": line_number,
                }
            )

        return entries, issues

    def _extract_entry_content(self, content: str, start: int) -> Optional[str]:
        """Extract content between braces starting at position."""
        brace_count = 1
        pos = start

        while pos < len(content) and brace_count > 0:
            if content[pos] == "{":
                brace_count += 1
            elif content[pos] == "}":
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            return content[start : pos - 1]
        return None

    def _parse_fields(self, content: str) -> dict:
        """Parse fields from entry content."""
        fields = {}

        # Pattern to match field = value
        # Handles: field = {value}, field = "value", field = number
        field_pattern = re.compile(
            r"(\w+)\s*=\s*(?:\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|\"([^\"]*)\"|(\d+))",
            re.IGNORECASE,
        )

        for match in field_pattern.finditer(content):
            field_name = match.group(1).lower()
            # Value is in group 2 (braces), 3 (quotes), or 4 (number)
            value = match.group(2) or match.group(3) or match.group(4) or ""
            fields[field_name] = value.strip()

        return fields

    def _validate_entry(self, entry: dict) -> List[ValidationIssue]:
        """Validate a single entry."""
        issues = []
        entry_type = entry.get("type", "")
        entry_key = entry.get("key", "")
        fields = entry.get("fields", {})
        line_number = entry.get("line_number")

        # Skip special entries
        if entry_type in ["string", "comment", "preamble"]:
            return issues

        # Check required fields
        required = REQUIRED_FIELDS.get(entry_type, [])
        for field in required:
            # Check for field or common alternatives
            has_field = field in fields
            # author can be satisfied by editor
            if field == "author" and "editor" in fields:
                has_field = True
            # title variations
            if field == "title" and not has_field:
                has_field = any(
                    f in fields for f in ["title", "booktitle", "chapter"]
                )

            if not has_field:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Missing recommended field: {field}",
                        line_number=line_number,
                        entry_key=entry_key,
                        field_name=field,
                    )
                )

        # Check for empty required fields
        for field_name, value in fields.items():
            if field_name in required and not value.strip():
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Empty value for field: {field_name}",
                        line_number=line_number,
                        entry_key=entry_key,
                        field_name=field_name,
                    )
                )

        # Validate year format
        year = fields.get("year", "")
        if year and not re.match(r"^\d{4}$", year):
            # Allow year ranges and approximate years
            if not re.match(r"^\d{4}[-/]\d{2,4}$|^circa\s*\d{4}$|^\d{4}\?$", year):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Non-standard year format: {year}",
                        line_number=line_number,
                        entry_key=entry_key,
                        field_name="year",
                    )
                )

        # Validate DOI format
        doi = fields.get("doi", "")
        if doi:
            # DOI should start with 10.
            if not doi.startswith("10.") and "doi.org" not in doi:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Invalid DOI format: {doi}",
                        line_number=line_number,
                        entry_key=entry_key,
                        field_name="doi",
                    )
                )

        return issues

    def validate_files(self, file_paths: List[str | Path]) -> List[ValidationResult]:
        """
        Validate multiple BibTeX files.

        Args:
            file_paths: List of paths to .bib files

        Returns:
            List of ValidationResults
        """
        return [self.validate_file(path) for path in file_paths]

    def validate_before_merge(
        self, file_paths: List[str | Path]
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate files before merging operation.

        Args:
            file_paths: List of paths to .bib files to merge

        Returns:
            Tuple of (can_merge, list of validation results)
        """
        results = self.validate_files(file_paths)
        can_merge = all(r.is_valid for r in results)

        # Check for cross-file duplicate keys
        all_keys: Set[str] = set()
        cross_file_duplicates: List[str] = []

        for result in results:
            # Get keys from this file's content
            if result.file_path:
                try:
                    content = Path(result.file_path).read_text()
                    entries, _ = self._parse_entries(content)
                    for entry in entries:
                        key = entry.get("key", "").lower()
                        if key:
                            if key in all_keys:
                                cross_file_duplicates.append(entry.get("key", ""))
                            all_keys.add(key)
                except Exception:
                    pass

        if cross_file_duplicates:
            can_merge = False
            # Add cross-file duplicate warning to first result
            if results:
                results[0].issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Cross-file duplicate keys found: {', '.join(cross_file_duplicates)}",
                    )
                )
                results[0].is_valid = False

        return can_merge, results


def validate_bibtex_file(file_path: str | Path, strict: bool = False) -> ValidationResult:
    """
    Convenience function to validate a single BibTeX file.

    Args:
        file_path: Path to the .bib file
        strict: If True, warnings are treated as errors

    Returns:
        ValidationResult
    """
    validator = BibTeXValidator(strict=strict)
    return validator.validate_file(file_path)


def validate_bibtex_content(content: str, strict: bool = False) -> ValidationResult:
    """
    Convenience function to validate BibTeX content string.

    Args:
        content: BibTeX content as string
        strict: If True, warnings are treated as errors

    Returns:
        ValidationResult
    """
    validator = BibTeXValidator(strict=strict)
    return validator.validate_content(content)


# EOF
