#!/usr/bin/env python3
# Timestamp: "2026-01-05 (ywatanabe)"
# File: tests/scitex/sh/test___init__.py

"""Tests for sh module public API.

This module tests the public interface of the sh module:
- sh: Main shell execution function
- sh_run: Convenience function for command execution
- quote: Re-exported from _security
"""

import pytest


class TestShFunctionBasic:
    """Basic tests for sh function."""

    def test_sh_import_from_module(self):
        """Test sh can be imported from scitex.sh."""
        from scitex.sh import sh

        assert sh is not None
        assert callable(sh)

    def test_sh_import_from_scitex(self):
        """Test sh can be imported from scitex.sh module."""
        import scitex.sh as sh_module

        assert hasattr(sh_module, "sh")
        assert callable(sh_module.sh)

    def test_sh_basic_command(self):
        """Test sh executes basic command."""
        from scitex.sh import sh

        result = sh(["echo", "hello"], verbose=False)

        assert isinstance(result, dict)
        assert result["success"] is True

    def test_sh_returns_dict_by_default(self):
        """Test sh returns dict by default."""
        from scitex.sh import sh

        result = sh(["echo", "test"], verbose=False)

        assert isinstance(result, dict)
        assert "stdout" in result
        assert "stderr" in result
        assert "exit_code" in result
        assert "success" in result

    def test_sh_return_as_dict(self):
        """Test sh returns dict when return_as='dict'."""
        from scitex.sh import sh

        result = sh(["echo", "test"], verbose=False, return_as="dict")

        assert isinstance(result, dict)
        assert result["stdout"] == "test"

    def test_sh_return_as_str_success(self):
        """Test sh returns stdout string when return_as='str' on success."""
        from scitex.sh import sh

        result = sh(["echo", "hello world"], verbose=False, return_as="str")

        assert isinstance(result, str)
        assert result == "hello world"

    def test_sh_return_as_str_failure(self):
        """Test sh returns stderr string when return_as='str' on failure."""
        from scitex.sh import sh

        result = sh(["cat", "/nonexistent/file"], verbose=False, return_as="str")

        assert isinstance(result, str)
        # stderr should contain error message
        assert len(result) > 0


class TestShFunctionOptions:
    """Tests for sh function options."""

    def test_sh_verbose_false(self):
        """Test sh with verbose=False."""
        from scitex.sh import sh

        # Should not raise and should work silently
        result = sh(["echo", "quiet"], verbose=False)
        assert result["success"] is True

    def test_sh_verbose_true(self, capsys):
        """Test sh with verbose=True prints output."""
        from scitex.sh import sh

        sh(["echo", "loud"], verbose=True)
        captured = capsys.readouterr()

        # Should print command and/or output
        assert "loud" in captured.out or "echo" in captured.out

    def test_sh_timeout_success(self):
        """Test sh with timeout that succeeds."""
        from scitex.sh import sh

        result = sh(["echo", "fast"], verbose=False, timeout=10)

        assert result["success"] is True

    def test_sh_timeout_exceeded(self):
        """Test sh with timeout that is exceeded."""
        from scitex.sh import sh

        result = sh(["sleep", "5"], verbose=False, timeout=1)

        assert result["success"] is False

    def test_sh_stream_output(self):
        """Test sh with stream_output enabled."""
        from scitex.sh import sh

        result = sh(
            ["echo", "streamed"],
            verbose=False,
            stream_output=True,
        )

        assert result["success"] is True
        assert "streamed" in result["stdout"]


class TestShFunctionSecurity:
    """Security tests for sh function."""

    def test_sh_rejects_string_command(self):
        """Test sh rejects string commands."""
        from scitex.sh import sh

        with pytest.raises(TypeError) as exc_info:
            sh("echo hello", verbose=False)

        assert "String commands are not allowed" in str(exc_info.value)

    def test_sh_rejects_null_byte(self):
        """Test sh rejects null byte in arguments."""
        from scitex.sh import sh

        with pytest.raises(ValueError):
            sh(["echo", "test\0malicious"], verbose=False)

    def test_sh_safe_with_special_chars_in_args(self):
        """Test sh safely handles special chars in arguments."""
        from scitex.sh import sh

        # These are dangerous as shell metacharacters but safe in list format
        result = sh(["echo", "test; rm -rf /"], verbose=False)

        assert result["success"] is True
        assert "test; rm -rf /" in result["stdout"]


class TestShRunFunction:
    """Tests for sh_run function."""

    def test_sh_run_import(self):
        """Test sh_run can be imported."""
        from scitex.sh import sh_run

        assert sh_run is not None
        assert callable(sh_run)

    def test_sh_run_basic(self):
        """Test sh_run executes basic command."""
        from scitex.sh import sh_run

        result = sh_run(["echo", "hello"], verbose=False)

        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["stdout"] == "hello"

    def test_sh_run_returns_shell_result(self):
        """Test sh_run returns ShellResult structure."""
        from scitex.sh import sh_run

        result = sh_run(["pwd"], verbose=False)

        assert "stdout" in result
        assert "stderr" in result
        assert "exit_code" in result
        assert "success" in result

    def test_sh_run_success(self):
        """Test sh_run with successful command."""
        from scitex.sh import sh_run

        result = sh_run(["true"], verbose=False)

        assert result["success"] is True
        assert result["exit_code"] == 0

    def test_sh_run_failure(self):
        """Test sh_run with failed command."""
        from scitex.sh import sh_run

        result = sh_run(["false"], verbose=False)

        assert result["success"] is False
        assert result["exit_code"] == 1

    def test_sh_run_captures_stderr(self):
        """Test sh_run captures stderr."""
        from scitex.sh import sh_run

        result = sh_run(["bash", "-c", "echo error >&2"], verbose=False)

        assert "error" in result["stderr"]

    def test_sh_run_verbose_option(self, capsys):
        """Test sh_run verbose option."""
        from scitex.sh import sh_run

        sh_run(["echo", "visible"], verbose=True)
        captured = capsys.readouterr()

        assert "visible" in captured.out or "echo" in captured.out


class TestQuoteExport:
    """Tests for quote function export."""

    def test_quote_import_from_sh(self):
        """Test quote can be imported from scitex.sh."""
        from scitex.sh import quote

        assert quote is not None
        assert callable(quote)

    def test_quote_basic(self):
        """Test quote basic functionality."""
        from scitex.sh import quote

        result = quote("test string")
        assert "test string" in result

    def test_quote_dangerous_string(self):
        """Test quote safely handles dangerous strings."""
        from scitex.sh import quote

        result = quote("test; rm -rf /")
        assert "'" in result


class TestModuleExports:
    """Tests for module exports and __all__."""

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        import scitex.sh as sh_module

        assert hasattr(sh_module, "__all__")
        assert "sh" in sh_module.__all__
        assert "sh_run" in sh_module.__all__
        assert "quote" in sh_module.__all__

    def test_sh_accessible_from_scitex(self):
        """Test sh is accessible from scitex namespace."""
        import scitex

        assert hasattr(scitex, "sh")

    def test_sh_run_accessible_from_scitex_sh(self):
        """Test sh_run is accessible from scitex.sh module."""
        import scitex.sh as sh_module

        assert hasattr(sh_module, "sh_run")
        assert callable(sh_module.sh_run)


class TestIntegration:
    """Integration tests for sh module."""

    def test_sh_and_sh_run_produce_same_result(self):
        """Test sh and sh_run produce equivalent results."""
        from scitex.sh import sh, sh_run

        sh_result = sh(["echo", "test"], verbose=False, return_as="dict")
        sh_run_result = sh_run(["echo", "test"], verbose=False)

        assert sh_result["stdout"] == sh_run_result["stdout"]
        assert sh_result["exit_code"] == sh_run_result["exit_code"]
        assert sh_result["success"] == sh_run_result["success"]

    def test_complete_workflow(self, tmp_path):
        """Test complete workflow with file operations."""
        from scitex.sh import sh, sh_run

        # Create a file
        test_file = tmp_path / "test.txt"
        sh_run(["touch", str(test_file)], verbose=False)
        assert test_file.exists()

        # Write content (using bash redirect in list format won't work,
        # so use echo and redirect manually)
        test_file.write_text("hello world")

        # Read file
        result = sh(["cat", str(test_file)], verbose=False)
        assert result["stdout"] == "hello world"

        # Count words
        result = sh_run(["wc", "-w", str(test_file)], verbose=False)
        assert result["success"] is True
        assert "2" in result["stdout"]

    def test_error_handling_workflow(self):
        """Test error handling workflow."""
        from scitex.sh import sh

        # Try to read nonexistent file
        result = sh(
            ["cat", "/nonexistent/file"],
            verbose=False,
            return_as="dict",
        )

        assert result["success"] is False
        assert result["exit_code"] != 0
        assert len(result["stderr"]) > 0


class TestShFunctionEdgeCases:
    """Edge case tests for sh function."""

    def test_sh_empty_output(self):
        """Test sh with command that produces no output."""
        from scitex.sh import sh

        result = sh(["true"], verbose=False)

        assert result["success"] is True
        assert result["stdout"] == ""

    def test_sh_multiline_output(self):
        """Test sh with multiline output."""
        from scitex.sh import sh

        result = sh(["printf", "line1\nline2\nline3"], verbose=False)

        assert result["success"] is True
        lines = result["stdout"].split("\n")
        assert len(lines) == 3

    def test_sh_unicode_output(self):
        """Test sh with unicode output."""
        from scitex.sh import sh

        result = sh(["echo", "日本語"], verbose=False)

        assert result["success"] is True
        assert "日本語" in result["stdout"]

    def test_sh_return_str_with_multiline(self):
        """Test sh return_as='str' with multiline output."""
        from scitex.sh import sh

        result = sh(
            ["printf", "a\nb\nc"],
            verbose=False,
            return_as="str",
        )

        assert isinstance(result, str)
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_sh_with_many_arguments(self):
        """Test sh with many arguments."""
        from scitex.sh import sh

        args = ["arg" + str(i) for i in range(100)]
        result = sh(["echo"] + args, verbose=False)

        assert result["success"] is True
        for arg in args:
            assert arg in result["stdout"]


class TestShRunEdgeCases:
    """Edge case tests for sh_run function."""

    def test_sh_run_empty_output(self):
        """Test sh_run with command that produces no output."""
        from scitex.sh import sh_run

        result = sh_run(["true"], verbose=False)

        assert result["success"] is True
        assert result["stdout"] == ""
        assert result["exit_code"] == 0

    def test_sh_run_with_path_argument(self):
        """Test sh_run with file path argument."""
        from scitex.sh import sh_run

        result = sh_run(["ls", "/tmp"], verbose=False)

        assert result["success"] is True

    def test_sh_run_rejects_string(self):
        """Test sh_run rejects string commands."""
        from scitex.sh import sh_run

        with pytest.raises(TypeError):
            sh_run("ls -la", verbose=False)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
