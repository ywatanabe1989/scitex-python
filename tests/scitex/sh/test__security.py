#!/usr/bin/env python3
# Timestamp: "2026-01-05 (ywatanabe)"
# File: tests/scitex/sh/test__security.py

"""Tests for shell command security functions.

This module tests the security functionality of the sh module:
- validate_command: Validates commands for security issues
- quote: Safely quotes arguments for shell use
- DANGEROUS_CHARS: List of dangerous shell characters
"""

import pytest


class TestValidateCommandBasic:
    """Basic tests for validate_command function."""

    def test_validate_command_import(self):
        """Test validate_command can be imported."""
        from scitex.sh._security import validate_command

        assert validate_command is not None
        assert callable(validate_command)

    def test_validate_command_accepts_list(self):
        """Test validate_command accepts list format."""
        from scitex.sh._security import validate_command

        # Should not raise any exception
        validate_command(["ls", "-la"])
        validate_command(["echo", "hello"])
        validate_command(["pwd"])

    def test_validate_command_rejects_string(self):
        """Test validate_command rejects string format."""
        from scitex.sh._security import validate_command

        with pytest.raises(TypeError) as exc_info:
            validate_command("ls -la")

        assert "String commands are not allowed" in str(exc_info.value)
        assert "security reasons" in str(exc_info.value)

    def test_validate_command_string_with_pipes_rejected(self):
        """Test string with pipes is rejected."""
        from scitex.sh._security import validate_command

        with pytest.raises(TypeError):
            validate_command("ls | grep test")

    def test_validate_command_string_with_semicolon_rejected(self):
        """Test string with semicolon is rejected."""
        from scitex.sh._security import validate_command

        with pytest.raises(TypeError):
            validate_command("ls; rm -rf /")


class TestValidateCommandNullByte:
    """Tests for null byte detection in validate_command."""

    def test_null_byte_in_argument_rejected(self):
        """Test null byte in argument is rejected."""
        from scitex.sh._security import validate_command

        with pytest.raises(ValueError) as exc_info:
            validate_command(["echo", "test\0malicious"])

        assert "null byte" in str(exc_info.value)
        assert "shell injection" in str(exc_info.value)

    def test_null_byte_in_command_rejected(self):
        """Test null byte in command is rejected."""
        from scitex.sh._security import validate_command

        with pytest.raises(ValueError):
            validate_command(["ls\0", "-la"])

    def test_null_byte_at_start_rejected(self):
        """Test null byte at start of argument is rejected."""
        from scitex.sh._security import validate_command

        with pytest.raises(ValueError):
            validate_command(["echo", "\0malicious"])

    def test_null_byte_at_end_rejected(self):
        """Test null byte at end of argument is rejected."""
        from scitex.sh._security import validate_command

        with pytest.raises(ValueError):
            validate_command(["echo", "test\0"])

    def test_multiple_null_bytes_rejected(self):
        """Test multiple null bytes are rejected."""
        from scitex.sh._security import validate_command

        with pytest.raises(ValueError):
            validate_command(["echo", "a\0b\0c"])


class TestValidateCommandValidCases:
    """Test validate_command with valid inputs."""

    def test_simple_command(self):
        """Test simple command passes validation."""
        from scitex.sh._security import validate_command

        # Should not raise
        validate_command(["ls"])

    def test_command_with_flags(self):
        """Test command with flags passes validation."""
        from scitex.sh._security import validate_command

        validate_command(["ls", "-la", "--color=auto"])

    def test_command_with_path(self):
        """Test command with path argument passes validation."""
        from scitex.sh._security import validate_command

        validate_command(["cat", "/etc/passwd"])
        validate_command(["ls", "/home/user/Documents"])

    def test_command_with_spaces_in_argument(self):
        """Test command with spaces in argument passes validation."""
        from scitex.sh._security import validate_command

        validate_command(["echo", "Hello World"])
        validate_command(["touch", "file with spaces.txt"])

    def test_command_with_special_chars_in_argument(self):
        """Test command with special chars as data passes validation."""
        from scitex.sh._security import validate_command

        # These are dangerous as shell metacharacters but safe in list format
        validate_command(["echo", "test; rm -rf /"])
        validate_command(["echo", "test | grep foo"])
        validate_command(["echo", "test && echo bar"])

    def test_command_with_unicode(self):
        """Test command with unicode characters passes validation."""
        from scitex.sh._security import validate_command

        validate_command(["echo", "日本語"])
        validate_command(["echo", "Привет мир"])

    def test_empty_argument_list(self):
        """Test empty argument is valid."""
        from scitex.sh._security import validate_command

        validate_command(["echo", ""])


class TestQuoteFunction:
    """Tests for the quote function."""

    def test_quote_import(self):
        """Test quote can be imported."""
        from scitex.sh._security import quote

        assert quote is not None
        assert callable(quote)

    def test_quote_simple_string(self):
        """Test quoting simple string."""
        from scitex.sh._security import quote

        result = quote("hello")
        assert result == "hello" or result == "'hello'"

    def test_quote_string_with_spaces(self):
        """Test quoting string with spaces."""
        from scitex.sh._security import quote

        result = quote("hello world")
        # shlex.quote wraps in single quotes
        assert result == "'hello world'"

    def test_quote_dangerous_string(self):
        """Test quoting dangerous string."""
        from scitex.sh._security import quote

        dangerous = "file; rm -rf /"
        result = quote(dangerous)
        # Should be safely quoted
        assert "'" in result or result.startswith("'")
        # The semicolon should be inside quotes, not as a command separator
        assert result != dangerous

    def test_quote_string_with_semicolon(self):
        """Test quoting string with semicolon."""
        from scitex.sh._security import quote

        result = quote("test;command")
        assert "'" in result

    def test_quote_string_with_pipe(self):
        """Test quoting string with pipe character."""
        from scitex.sh._security import quote

        result = quote("test|command")
        assert "'" in result

    def test_quote_string_with_ampersand(self):
        """Test quoting string with ampersand."""
        from scitex.sh._security import quote

        result = quote("test&command")
        assert "'" in result

    def test_quote_string_with_dollar(self):
        """Test quoting string with dollar sign."""
        from scitex.sh._security import quote

        result = quote("$HOME")
        assert "'" in result

    def test_quote_string_with_backtick(self):
        """Test quoting string with backtick."""
        from scitex.sh._security import quote

        result = quote("`whoami`")
        # Backticks should be quoted
        assert "'" in result

    def test_quote_empty_string(self):
        """Test quoting empty string."""
        from scitex.sh._security import quote

        result = quote("")
        assert result == "''"

    def test_quote_string_with_single_quotes(self):
        """Test quoting string containing single quotes."""
        from scitex.sh._security import quote

        result = quote("it's a test")
        # Should handle single quotes properly
        assert "it" in result
        assert "a test" in result

    def test_quote_string_with_double_quotes(self):
        """Test quoting string containing double quotes."""
        from scitex.sh._security import quote

        result = quote('say "hello"')
        assert "hello" in result

    def test_quote_preserves_content(self):
        """Test that quote preserves the original content."""
        import shlex

        from scitex.sh._security import quote

        original = "test data 123"
        quoted = quote(original)
        # When unquoted via shell, should get original back
        # shlex.split can unquote for us
        unquoted = shlex.split(quoted)[0]
        assert unquoted == original

    def test_quote_newline(self):
        """Test quoting string with newline."""
        from scitex.sh._security import quote

        result = quote("line1\nline2")
        # Newline should be inside quotes
        assert "\n" in result or "'" in result

    def test_quote_tab(self):
        """Test quoting string with tab."""
        from scitex.sh._security import quote

        result = quote("col1\tcol2")
        assert "\t" in result or "'" in result


class TestDangerousChars:
    """Tests for DANGEROUS_CHARS constant."""

    def test_dangerous_chars_import(self):
        """Test DANGEROUS_CHARS can be imported."""
        from scitex.sh._security import DANGEROUS_CHARS

        assert DANGEROUS_CHARS is not None
        assert isinstance(DANGEROUS_CHARS, list)

    def test_dangerous_chars_contains_semicolon(self):
        """Test DANGEROUS_CHARS contains semicolon."""
        from scitex.sh._security import DANGEROUS_CHARS

        assert ";" in DANGEROUS_CHARS

    def test_dangerous_chars_contains_pipe(self):
        """Test DANGEROUS_CHARS contains pipe."""
        from scitex.sh._security import DANGEROUS_CHARS

        assert "|" in DANGEROUS_CHARS

    def test_dangerous_chars_contains_ampersand(self):
        """Test DANGEROUS_CHARS contains ampersand."""
        from scitex.sh._security import DANGEROUS_CHARS

        assert "&" in DANGEROUS_CHARS

    def test_dangerous_chars_contains_dollar(self):
        """Test DANGEROUS_CHARS contains dollar sign."""
        from scitex.sh._security import DANGEROUS_CHARS

        assert "$" in DANGEROUS_CHARS

    def test_dangerous_chars_contains_backtick(self):
        """Test DANGEROUS_CHARS contains backtick."""
        from scitex.sh._security import DANGEROUS_CHARS

        assert "`" in DANGEROUS_CHARS

    def test_dangerous_chars_contains_newline(self):
        """Test DANGEROUS_CHARS contains newline."""
        from scitex.sh._security import DANGEROUS_CHARS

        assert "\n" in DANGEROUS_CHARS

    def test_dangerous_chars_contains_redirects(self):
        """Test DANGEROUS_CHARS contains redirect characters."""
        from scitex.sh._security import DANGEROUS_CHARS

        assert ">" in DANGEROUS_CHARS
        assert "<" in DANGEROUS_CHARS

    def test_dangerous_chars_contains_parentheses(self):
        """Test DANGEROUS_CHARS contains parentheses."""
        from scitex.sh._security import DANGEROUS_CHARS

        assert "(" in DANGEROUS_CHARS
        assert ")" in DANGEROUS_CHARS

    def test_dangerous_chars_contains_braces(self):
        """Test DANGEROUS_CHARS contains braces."""
        from scitex.sh._security import DANGEROUS_CHARS

        assert "{" in DANGEROUS_CHARS
        assert "}" in DANGEROUS_CHARS


class TestSecurityIntegration:
    """Integration tests for security functions."""

    def test_quote_makes_dangerous_string_safe(self):
        """Test that quote makes dangerous strings safe."""
        from scitex.sh._security import DANGEROUS_CHARS, quote

        for char in DANGEROUS_CHARS:
            dangerous_str = f"test{char}malicious"
            quoted = quote(dangerous_str)
            # The quoted string should be wrapped in quotes
            assert "'" in quoted or quoted.startswith("'")

    def test_validate_accepts_quoted_dangerous_args(self):
        """Test validate_command accepts safely quoted dangerous content."""
        from scitex.sh._security import quote, validate_command

        # These are safe because they're in list format - the content is data
        validate_command(["echo", "test; rm -rf /"])
        validate_command(["echo", "$(whoami)"])
        validate_command(["cat", "file`id`"])

    def test_error_messages_are_informative(self):
        """Test that error messages provide guidance."""
        from scitex.sh._security import validate_command

        with pytest.raises(TypeError) as exc_info:
            validate_command("ls -la")

        error_msg = str(exc_info.value)
        # Should mention list format
        assert "list" in error_msg.lower()
        # Should mention security
        assert "security" in error_msg.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_validate_very_long_argument_list(self):
        """Test validation of very long argument list."""
        from scitex.sh._security import validate_command

        # Create a command with many arguments
        cmd = ["echo"] + [f"arg{i}" for i in range(1000)]
        validate_command(cmd)  # Should not raise

    def test_validate_empty_list(self):
        """Test validation of empty list."""
        from scitex.sh._security import validate_command

        # Empty list might be valid (no command)
        # Implementation behavior may vary
        try:
            validate_command([])
        except (ValueError, IndexError):
            pass  # Some implementations may reject empty lists

    def test_quote_very_long_string(self):
        """Test quoting very long string."""
        from scitex.sh._security import quote

        long_str = "a" * 10000
        result = quote(long_str)
        assert len(result) >= len(long_str)

    def test_quote_unicode_special_chars(self):
        """Test quoting unicode special characters."""
        from scitex.sh._security import quote

        # Various unicode that might cause issues
        unicode_strings = [
            "日本語",
            "Ñoño",
            "émoji 🎉",
            "αβγ",
            "中文测试",
        ]

        for s in unicode_strings:
            result = quote(s)
            assert s in result or "'" in result


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
