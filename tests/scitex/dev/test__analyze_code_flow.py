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
    def should_not_appear():
        pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            analyzer = CodeFlowAnalyzer(temp_path)
            result = analyzer.analyze()
            assert "real_function" in result
            assert "should_not_appear" not in result
        finally:
            os.unlink(temp_path)

    def test_analyze_nonexistent_file(self):
        """Test analyzing a nonexistent file returns error."""
        analyzer = CodeFlowAnalyzer("/nonexistent/path/file.py")
        result = analyzer.analyze()
        # Should return error string
        assert isinstance(result, str)

    def test_analyze_empty_file(self):
        """Test analyzing an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            analyzer = CodeFlowAnalyzer(temp_path)
            result = analyzer.analyze()
            assert "Execution Flow:" in result
        finally:
            os.unlink(temp_path)

    def test_analyze_with_nested_calls(self):
        """Test analyzing nested function calls."""
        code = """
def outer():
    def inner():
        pass
    inner()

outer()
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            analyzer = CodeFlowAnalyzer(temp_path)
            result = analyzer.analyze()
            assert "outer" in result
            assert "inner" in result
        finally:
            os.unlink(temp_path)

    def test_analyze_with_attribute_calls(self):
        """Test analyzing attribute-based function calls."""
        code = """
def process():
    obj = MyClass()
    obj.custom_method()
    return obj
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            analyzer = CodeFlowAnalyzer(temp_path)
            result = analyzer.analyze()
            assert "process" in result
            # MyClass should be tracked as a function call
            assert "MyClass" in result
        finally:
            os.unlink(temp_path)


class TestAnalyzeCodeFlow:
    """Tests for the analyze_code_flow convenience function."""

    def test_analyze_code_flow_returns_string(self):
        """Test that analyze_code_flow returns a string result."""
        code = """
def test_func():
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

    def test_analyze_code_flow_with_none_path(self):
        """Test analyze_code_flow with None path."""
        analyzer = CodeFlowAnalyzer(None)
        result = analyzer.analyze()
        # Should return None when file_path is None
        assert result is None


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
