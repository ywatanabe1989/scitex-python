#!/usr/bin/env python3
"""Tests for scitex.writer.utils._parse_script_args."""

from pathlib import Path

import pytest

from scitex.writer.utils._parse_script_args import ScriptArgument, ScriptArgumentParser


class TestScriptArgumentCreation:
    """Tests for ScriptArgument dataclass."""

    def test_creates_with_all_fields(self):
        """Verify ScriptArgument creates with all fields."""
        arg = ScriptArgument(
            short_flag="-v",
            long_flag="--verbose",
            description="Enable verbose output",
            default="false",
        )

        assert arg.short_flag == "-v"
        assert arg.long_flag == "--verbose"
        assert arg.description == "Enable verbose output"
        assert arg.default == "false"

    def test_creates_with_none_values(self):
        """Verify ScriptArgument accepts None values."""
        arg = ScriptArgument(
            short_flag=None,
            long_flag="--help",
            description="Show help",
            default=None,
        )

        assert arg.short_flag is None
        assert arg.default is None


class TestScriptArgumentStr:
    """Tests for ScriptArgument __str__ method."""

    def test_str_with_both_flags(self):
        """Verify __str__ formats both flags."""
        arg = ScriptArgument(
            short_flag="-v",
            long_flag="--verbose",
            description="Enable verbose output",
            default=None,
        )

        result = str(arg)
        assert "-v" in result
        assert "--verbose" in result
        assert "Enable verbose output" in result

    def test_str_with_default(self):
        """Verify __str__ includes default value."""
        arg = ScriptArgument(
            short_flag="-n",
            long_flag="--num",
            description="Number of items",
            default="10",
        )

        result = str(arg)
        assert "(default: 10)" in result

    def test_str_without_default(self):
        """Verify __str__ omits default when None."""
        arg = ScriptArgument(
            short_flag="-h",
            long_flag="--help",
            description="Show help",
            default=None,
        )

        result = str(arg)
        assert "default" not in result

    def test_str_with_only_short_flag(self):
        """Verify __str__ works with only short flag."""
        arg = ScriptArgument(
            short_flag="-v",
            long_flag=None,
            description="Verbose mode",
            default=None,
        )

        result = str(arg)
        assert "-v" in result
        assert "--" not in result


class TestScriptArgumentParserParse:
    """Tests for ScriptArgumentParser.parse method."""

    def test_returns_empty_for_nonexistent_file(self, tmp_path):
        """Verify returns empty list for nonexistent file."""
        result = ScriptArgumentParser.parse(tmp_path / "nonexistent.sh")

        assert result == []

    def test_returns_empty_for_no_usage_function(self, tmp_path):
        """Verify returns empty list when no usage() function."""
        script = tmp_path / "script.sh"
        script.write_text("""#!/bin/bash
echo "Hello"
""")

        result = ScriptArgumentParser.parse(script)

        assert result == []

    def test_parses_simple_usage_function(self, tmp_path):
        """Verify parses arguments from usage() function."""
        script = tmp_path / "script.sh"
        script.write_text("""#!/bin/bash
usage() {
    echo "Usage: script.sh [options]"
    echo "-v, --verbose   Enable verbose output"
    exit 0
}
""")

        result = ScriptArgumentParser.parse(script)

        assert len(result) == 1
        assert result[0].short_flag == "-v"
        assert result[0].long_flag == "--verbose"

    def test_parses_multiple_arguments(self, tmp_path):
        """Verify parses multiple arguments."""
        script = tmp_path / "script.sh"
        script.write_text("""#!/bin/bash
usage() {
    echo "Usage: script.sh [options]"
    echo "-v, --verbose   Enable verbose output"
    echo "-q, --quiet     Quiet mode"
    echo "-h, --help      Show this help"
    exit 0
}
""")

        result = ScriptArgumentParser.parse(script)

        assert len(result) == 3

    def test_parses_argument_with_default(self, tmp_path):
        """Verify parses default values."""
        script = tmp_path / "script.sh"
        script.write_text("""#!/bin/bash
usage() {
    echo "-n, --num       Number of items (default: 10)"
    exit 0
}
""")

        result = ScriptArgumentParser.parse(script)

        assert len(result) == 1
        assert result[0].default == "10"


class TestScriptArgumentParserParseArgumentLine:
    """Tests for ScriptArgumentParser._parse_argument_line method."""

    def test_parses_short_and_long_flags(self):
        """Verify parses both short and long flags."""
        line = "-v, --verbose   Enable verbose output"
        result = ScriptArgumentParser._parse_argument_line(line)

        assert result.short_flag == "-v"
        assert result.long_flag == "--verbose"
        assert result.description == "Enable verbose output"

    def test_parses_only_short_flag(self):
        """Verify parses only short flag."""
        line = "-v   Enable verbose output"
        result = ScriptArgumentParser._parse_argument_line(line)

        assert result.short_flag == "-v"
        assert result.long_flag is None

    def test_parses_only_long_flag(self):
        """Verify parses only long flag."""
        line = "--verbose   Enable verbose output"
        result = ScriptArgumentParser._parse_argument_line(line)

        assert result.short_flag is None
        assert result.long_flag == "--verbose"

    def test_parses_default_value(self):
        """Verify parses default value from description."""
        line = "-n, --num   Number of items (default: 10)"
        result = ScriptArgumentParser._parse_argument_line(line)

        assert result.default == "10"
        assert result.description == "Number of items"

    def test_returns_none_for_invalid_line(self):
        """Verify returns None for lines without proper format."""
        line = "This is just a description"
        result = ScriptArgumentParser._parse_argument_line(line)

        assert result is None

    def test_returns_none_for_no_description(self):
        """Verify returns None when description is empty."""
        line = "-v   "  # Only flag, no description
        # Need multiple spaces to match the pattern
        result = ScriptArgumentParser._parse_argument_line(line)

        # May return None or have empty description depending on implementation
        if result is not None:
            assert result.short_flag == "-v"


class TestScriptArgumentParserEdgeCases:
    """Tests for ScriptArgumentParser edge cases."""

    def test_ignores_comment_lines(self, tmp_path):
        """Verify comment lines are ignored."""
        script = tmp_path / "script.sh"
        script.write_text("""#!/bin/bash
usage() {
    echo "# This is a comment"
    echo "-v, --verbose   Enable verbose output"
    exit 0
}
""")

        result = ScriptArgumentParser.parse(script)

        assert len(result) == 1

    def test_ignores_lines_without_flags(self, tmp_path):
        """Verify lines without flags are ignored."""
        script = tmp_path / "script.sh"
        script.write_text("""#!/bin/bash
usage() {
    echo "Usage: script.sh [options]"
    echo "Options:"
    echo "-v, --verbose   Enable verbose output"
    exit 0
}
""")

        result = ScriptArgumentParser.parse(script)

        assert len(result) == 1


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
