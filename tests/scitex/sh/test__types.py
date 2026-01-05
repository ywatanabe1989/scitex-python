#!/usr/bin/env python3
# Timestamp: "2026-01-05 (ywatanabe)"
# File: tests/scitex/sh/test__types.py

"""Tests for shell command type definitions.

This module tests the type definitions used in the sh module:
- ShellResult TypedDict for command execution results
- CommandInput type alias for command arguments
- ReturnFormat literal type for return format specification
"""

from typing import get_args, get_origin, get_type_hints

import pytest


class TestShellResult:
    """Test ShellResult TypedDict structure."""

    def test_shell_result_import(self):
        """Test ShellResult can be imported."""
        from scitex.sh._types import ShellResult

        assert ShellResult is not None

    def test_shell_result_is_typed_dict(self):
        """Test ShellResult is a TypedDict."""
        from typing import TypedDict

        from scitex.sh._types import ShellResult

        # TypedDict classes have __annotations__
        assert hasattr(ShellResult, "__annotations__")
        # Check it has the expected structure
        annotations = ShellResult.__annotations__
        assert "stdout" in annotations
        assert "stderr" in annotations
        assert "exit_code" in annotations
        assert "success" in annotations

    def test_shell_result_field_types(self):
        """Test ShellResult has correct field types."""
        from scitex.sh._types import ShellResult

        # Use get_type_hints to resolve forward references from __future__ annotations
        hints = get_type_hints(ShellResult)
        assert hints["stdout"] == str
        assert hints["stderr"] == str
        assert hints["exit_code"] == int
        assert hints["success"] == bool

    def test_shell_result_can_be_instantiated(self):
        """Test ShellResult dict can be created with correct structure."""
        from scitex.sh._types import ShellResult

        result: ShellResult = {
            "stdout": "output text",
            "stderr": "",
            "exit_code": 0,
            "success": True,
        }

        assert result["stdout"] == "output text"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0
        assert result["success"] is True

    def test_shell_result_success_case(self):
        """Test ShellResult for successful command."""
        from scitex.sh._types import ShellResult

        result: ShellResult = {
            "stdout": "Hello World",
            "stderr": "",
            "exit_code": 0,
            "success": True,
        }

        assert result["success"] is True
        assert result["exit_code"] == 0

    def test_shell_result_failure_case(self):
        """Test ShellResult for failed command."""
        from scitex.sh._types import ShellResult

        result: ShellResult = {
            "stdout": "",
            "stderr": "File not found",
            "exit_code": 1,
            "success": False,
        }

        assert result["success"] is False
        assert result["exit_code"] == 1
        assert "File not found" in result["stderr"]

    def test_shell_result_multiline_output(self):
        """Test ShellResult with multiline output."""
        from scitex.sh._types import ShellResult

        multiline_output = "line1\nline2\nline3"
        result: ShellResult = {
            "stdout": multiline_output,
            "stderr": "",
            "exit_code": 0,
            "success": True,
        }

        assert result["stdout"].count("\n") == 2

    def test_shell_result_unicode_content(self):
        """Test ShellResult with Unicode content."""
        from scitex.sh._types import ShellResult

        unicode_output = "日本語テスト 中文测试 한국어테스트"
        result: ShellResult = {
            "stdout": unicode_output,
            "stderr": "",
            "exit_code": 0,
            "success": True,
        }

        assert result["stdout"] == unicode_output


class TestCommandInput:
    """Test CommandInput type alias."""

    def test_command_input_import(self):
        """Test CommandInput can be imported."""
        from scitex.sh._types import CommandInput

        assert CommandInput is not None

    def test_command_input_is_list_of_strings(self):
        """Test CommandInput is List[str]."""
        from typing import List

        from scitex.sh._types import CommandInput

        # CommandInput should be List[str]
        assert get_origin(CommandInput) == list
        args = get_args(CommandInput)
        assert args == (str,)

    def test_command_input_valid_examples(self):
        """Test valid CommandInput examples."""
        from scitex.sh._types import CommandInput

        # These are valid CommandInput values
        cmd1: CommandInput = ["ls", "-la"]
        cmd2: CommandInput = ["echo", "Hello World"]
        cmd3: CommandInput = ["git", "commit", "-m", "message"]
        cmd4: CommandInput = ["python", "-c", 'print("test")']

        assert isinstance(cmd1, list)
        assert all(isinstance(arg, str) for arg in cmd1)
        assert isinstance(cmd2, list)
        assert isinstance(cmd3, list)
        assert isinstance(cmd4, list)

    def test_command_input_single_command(self):
        """Test CommandInput with single command."""
        from scitex.sh._types import CommandInput

        cmd: CommandInput = ["pwd"]
        assert len(cmd) == 1
        assert cmd[0] == "pwd"


class TestReturnFormat:
    """Test ReturnFormat literal type."""

    def test_return_format_import(self):
        """Test ReturnFormat can be imported."""
        from scitex.sh._types import ReturnFormat

        assert ReturnFormat is not None

    def test_return_format_is_literal(self):
        """Test ReturnFormat is a Literal type."""
        from typing import Literal

        from scitex.sh._types import ReturnFormat

        # ReturnFormat should be Literal["dict", "str"]
        assert get_origin(ReturnFormat) == Literal

    def test_return_format_allowed_values(self):
        """Test ReturnFormat allowed values."""
        from scitex.sh._types import ReturnFormat

        allowed = get_args(ReturnFormat)
        assert "dict" in allowed
        assert "str" in allowed
        assert len(allowed) == 2

    def test_return_format_dict_value(self):
        """Test 'dict' is valid ReturnFormat."""
        from scitex.sh._types import ReturnFormat

        format_type: ReturnFormat = "dict"
        assert format_type == "dict"

    def test_return_format_str_value(self):
        """Test 'str' is valid ReturnFormat."""
        from scitex.sh._types import ReturnFormat

        format_type: ReturnFormat = "str"
        assert format_type == "str"


class TestModuleExports:
    """Test module exports and accessibility."""

    def test_all_types_importable_from_module(self):
        """Test all types can be imported from _types module."""
        from scitex.sh._types import CommandInput, ReturnFormat, ShellResult

        assert ShellResult is not None
        assert CommandInput is not None
        assert ReturnFormat is not None

    def test_types_consistency(self):
        """Test types are consistent with usage patterns."""
        from scitex.sh._types import CommandInput, ReturnFormat, ShellResult

        # Create a mock scenario
        command: CommandInput = ["echo", "test"]
        return_format: ReturnFormat = "dict"
        result: ShellResult = {
            "stdout": "test",
            "stderr": "",
            "exit_code": 0,
            "success": True,
        }

        # All should work without type errors
        assert isinstance(command, list)
        assert return_format in ("dict", "str")
        assert isinstance(result, dict)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
