#!/usr/bin/env python3
# Time-stamp: "2026-01-04 21:00:00 (ywatanabe)"
# File: ./tests/scitex/utils/test__verify_scitex_format_v01.py

"""
Functionality:
    * Tests SciTeX template format verification v01 functionality
    * Validates document parsing and compliance checking
    * Tests report generation from document text
Input:
    * Test document text and Python file content
Output:
    * Test results
Prerequisites:
    * pytest
"""

import pytest

from scitex.utils._verify_scitex_format_v01 import (
    FileInfo,
    TemplateCompliance,
    check_compliance,
    generate_report,
    parse_document,
)


class TestTemplateComplianceV01:
    """Test cases for TemplateCompliance dataclass v01."""

    def test_compliance_default_values(self):
        """Test default values of TemplateCompliance."""
        compliance = TemplateCompliance()

        assert compliance.has_main is False
        assert compliance.has_parse_args is False
        assert compliance.has_run_main is False
        assert compliance.has_main_guard is False
        assert compliance.has_docstring is False
        assert compliance.has_imports_section is False
        assert compliance.has_functions_classes_section is False
        assert compliance.uses_scitex_session is False
        assert compliance.has_verbose_param is False

    def test_is_compliant_all_required(self):
        """Test is_compliant property when all required items are present."""
        compliance = TemplateCompliance(
            has_main=True,
            has_parse_args=True,
            has_run_main=True,
            has_main_guard=True,
        )

        assert compliance.is_compliant is True

    def test_is_compliant_missing_required(self):
        """Test is_compliant property when required items are missing."""
        compliance = TemplateCompliance(
            has_main=True,
            has_parse_args=True,
            has_run_main=True,
            has_main_guard=False,  # Missing
        )

        assert compliance.is_compliant is False

    def test_compliance_score_all_true(self):
        """Test compliance_score when all items are True."""
        compliance = TemplateCompliance(
            has_main=True,
            has_parse_args=True,
            has_run_main=True,
            has_main_guard=True,
            has_docstring=True,
            has_imports_section=True,
            has_functions_classes_section=True,
            uses_scitex_session=True,
            has_verbose_param=True,
        )

        assert compliance.compliance_score == 1.0

    def test_compliance_score_all_false(self):
        """Test compliance_score when all items are False."""
        compliance = TemplateCompliance()

        assert compliance.compliance_score == 0.0

    def test_compliance_score_partial(self):
        """Test compliance_score with partial compliance."""
        compliance = TemplateCompliance(
            has_main=True,
            has_parse_args=True,
            has_run_main=True,
            has_main_guard=True,
        )
        # 4 out of 9 items are True
        expected_score = 4 / 9
        assert abs(compliance.compliance_score - expected_score) < 0.001


class TestFileInfoV01:
    """Test cases for FileInfo dataclass v01."""

    def test_file_info_creation(self):
        """Test FileInfo creation."""
        info = FileInfo(
            relative_path="test/file.py", content="print('hello')", size=100, lines=5
        )

        assert info.relative_path == "test/file.py"
        assert info.content == "print('hello')"
        assert info.size == 100
        assert info.lines == 5


class TestCheckComplianceV01:
    """Test cases for check_compliance function v01."""

    def test_check_compliance_empty_content(self):
        """Test check_compliance with empty content."""
        compliance = check_compliance("")

        assert compliance.has_main is False
        assert compliance.has_parse_args is False
        assert compliance.has_run_main is False
        assert compliance.has_main_guard is False

    def test_check_compliance_has_main(self):
        """Test check_compliance detects main function."""
        content = """
def main(args):
    pass
"""
        compliance = check_compliance(content)

        assert compliance.has_main is True

    def test_check_compliance_has_parse_args(self):
        """Test check_compliance detects parse_args function."""
        content = """
def parse_args():
    pass
"""
        compliance = check_compliance(content)

        assert compliance.has_parse_args is True

    def test_check_compliance_has_run_main(self):
        """Test check_compliance detects run_main function."""
        content = """
def run_main():
    pass
"""
        compliance = check_compliance(content)

        assert compliance.has_run_main is True

    def test_check_compliance_has_main_guard(self):
        """Test check_compliance detects main guard."""
        content = """
if __name__ == "__main__":
    pass
"""
        compliance = check_compliance(content)

        assert compliance.has_main_guard is True

    def test_check_compliance_full_template(self):
        """Test check_compliance with a full template."""
        content = """
def main(args):
    pass

def parse_args():
    pass

def run_main():
    pass

if __name__ == "__main__":
    run_main()
"""
        compliance = check_compliance(content)

        assert compliance.has_main is True
        assert compliance.has_parse_args is True
        assert compliance.has_run_main is True
        assert compliance.has_main_guard is True
        assert compliance.is_compliant is True


class TestParseDocumentV01:
    """Test cases for parse_document function v01."""

    def test_parse_document_empty(self):
        """Test parse_document with empty string."""
        files = parse_document("")

        assert files == {}

    def test_parse_document_no_py_files(self):
        """Test parse_document with no .py files."""
        doc_text = "Some random text without file markers"
        files = parse_document(doc_text)

        assert files == {}


class TestGenerateReportV01:
    """Test cases for generate_report function v01."""

    def test_generate_report_empty(self):
        """Test generate_report with empty results raises ZeroDivisionError."""
        results = {}
        # Note: generate_report has a bug - division by zero when no files
        with pytest.raises(ZeroDivisionError):
            generate_report(results)

    def test_generate_report_with_results(self):
        """Test generate_report with some results."""
        file_info = FileInfo(
            relative_path="test.py", content="def main(): pass", size=20, lines=1
        )
        compliance = TemplateCompliance(has_main=True)
        results = {"test.py": (file_info, compliance)}

        report = generate_report(results)

        assert "TEMPLATE COMPLIANCE REPORT" in report
        assert "Total Python files analyzed: 1" in report
        assert "test.py" in report


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
