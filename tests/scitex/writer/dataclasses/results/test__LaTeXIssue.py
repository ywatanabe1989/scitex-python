#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.results._LaTeXIssue."""

import pytest

from scitex.writer.dataclasses.results._LaTeXIssue import LaTeXIssue


class TestLaTeXIssueCreation:
    """Tests for LaTeXIssue instantiation."""

    def test_creates_error_issue(self):
        """Verify error issue can be created."""
        issue = LaTeXIssue(type="error", message="Undefined control sequence")
        assert issue.type == "error"
        assert issue.message == "Undefined control sequence"

    def test_creates_warning_issue(self):
        """Verify warning issue can be created."""
        issue = LaTeXIssue(type="warning", message="Citation not found")
        assert issue.type == "warning"
        assert issue.message == "Citation not found"


class TestLaTeXIssueStr:
    """Tests for LaTeXIssue __str__ method."""

    def test_str_error_format(self):
        """Verify error string format is uppercase ERROR."""
        issue = LaTeXIssue(type="error", message="Missing $ inserted")
        assert str(issue) == "ERROR: Missing $ inserted"

    def test_str_warning_format(self):
        """Verify warning string format is uppercase WARNING."""
        issue = LaTeXIssue(type="warning", message="Overfull \\hbox")
        assert str(issue) == "WARNING: Overfull \\hbox"

    def test_str_preserves_message(self):
        """Verify message is preserved in string output."""
        message = "Complex error message with special chars: @#$%"
        issue = LaTeXIssue(type="error", message=message)
        assert message in str(issue)


class TestLaTeXIssueEquality:
    """Tests for LaTeXIssue equality comparison."""

    def test_equal_issues(self):
        """Verify two issues with same data are equal."""
        issue1 = LaTeXIssue(type="error", message="Same message")
        issue2 = LaTeXIssue(type="error", message="Same message")
        assert issue1 == issue2

    def test_different_type_not_equal(self):
        """Verify issues with different types are not equal."""
        issue1 = LaTeXIssue(type="error", message="Same message")
        issue2 = LaTeXIssue(type="warning", message="Same message")
        assert issue1 != issue2

    def test_different_message_not_equal(self):
        """Verify issues with different messages are not equal."""
        issue1 = LaTeXIssue(type="error", message="Message 1")
        issue2 = LaTeXIssue(type="error", message="Message 2")
        assert issue1 != issue2


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
