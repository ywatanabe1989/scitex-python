#!/usr/bin/env python3
"""Tests for scitex.dev._analyze_code_flow module."""

import os
import tempfile
from pathlib import Path

import pytest

from scitex.dev._analyze_code_flow import CodeFlowAnalyzer, analyze_code_flow


class TestCodeFlowAnalyzer:
    """Tests for the CodeFlowAnalyzer class."""

    def test_init_creates_analyzer(self):
        """Test that CodeFlowAnalyzer initializes correctly."""
        analyzer = CodeFlowAnalyzer("test.py")
        assert analyzer.file_path == "test.py"
        assert analyzer.execution_flow == []
        assert analyzer.sequence == 1
        assert isinstance(analyzer.skip_functions, set)

    def test_skip_functions_contains_builtins(self):
        """Test that skip_functions includes Python builtins."""
        analyzer = CodeFlowAnalyzer("test.py")
        builtins = {"len", "min", "max", "sum", "print", "str", "int", "float"}
        assert builtins.issubset(analyzer.skip_functions)

    def test_analyze_simple_function(self):
        """Test analyzing a file with simple function definition."""
        code = """
def hello():
    return "Hello"

def greet(name):
    message = hello()
    return f"{message}, {name}"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            analyzer = CodeFlowAnalyzer(temp_path)
            result = analyzer.analyze()
            assert "Execution Flow:" in result
            assert "hello" in result
            assert "greet" in result
        finally:
            os.unlink(temp_path)

    def test_analyze_with_class(self):
        """Test analyzing a file with class definition."""
        code = """
class MyClass:
    def method_one(self):
        pass

    def method_two(self):
        self.method_one()
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            analyzer = CodeFlowAnalyzer(temp_path)
            result = analyzer.analyze()
            assert "Execution Flow:" in result
            assert "MyClass" in result
        finally:
            os.unlink(temp_path)

    def test_analyze_skips_main_guard(self):
        """Test that analyzer skips content after if __name__ == '__main__'."""
        code = """
def real_function():
    pass

if __name__ == "__main__":
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            analyzer = CodeFlowAnalyzer(temp_path)
            result = analyzer.analyze()
            assert "Execution Flow:" in result
            assert "real_function" in result
        finally:
            os.unlink(temp_path)


class TestAnalyzeCodeFlowFunction:
    """Tests for the analyze_code_flow function."""

    def test_analyze_code_flow_returns_string(self):
        """Test that analyze_code_flow returns a string."""
        code = """
def foo():
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = analyze_code_flow(temp_path)
            assert isinstance(result, str)
            assert "Execution Flow:" in result
        finally:
            os.unlink(temp_path)

    def test_analyze_code_flow_nonexistent_file(self):
        """Test analyze_code_flow with non-existent file."""
        result = analyze_code_flow("/nonexistent/file.py")
        assert isinstance(result, str)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
