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
    run_main()
"""
        compliance = check_compliance(content)

        assert compliance.has_main_guard is True

    def test_check_compliance_has_docstring(self):
        """Test check_compliance detects docstring."""
        content = '''
"""This is a module docstring."""
'''
        compliance = check_compliance(content)

        assert compliance.has_docstring is True

    def test_check_compliance_has_imports_section(self):
        """Test check_compliance detects Imports section."""
        content = '''
"""Imports"""
import os
'''
        compliance = check_compliance(content)

        assert compliance.has_imports_section is True

    def test_check_compliance_has_functions_classes_section(self):
        """Test check_compliance detects Functions & Classes section."""
        content = '''
"""Functions & Classes"""
def foo():
    pass
'''
        compliance = check_compliance(content)

        assert compliance.has_functions_classes_section is True

    def test_check_compliance_uses_scitex_session(self):
        """Test check_compliance detects scitex session usage."""
        content = """
CONFIG = stx.session.start()
"""
        compliance = check_compliance(content)

        assert compliance.uses_scitex_session is True

    def test_check_compliance_has_verbose_param(self):
        """Test check_compliance detects verbose parameter."""
        content = """
verbose: bool = True
verbose=False
"""
        compliance = check_compliance(content)

        assert compliance.has_verbose_param is True


class TestParseDocument:
    """Test cases for parse_document function."""

    def test_parse_document_empty(self):
        """Test parse_document with empty document."""
        files = parse_document("")

        assert len(files) == 0

    def test_parse_document_single_file(self):
        """Test parse_document with a single Python file."""
        doc_text = """
================================================================================
File: /path/to/script.py
Relative: script.py
Size: 100 bytes, Lines: 10, Modified: 2024-01-01
================================================================================
def main():
    pass
"""
        files = parse_document(doc_text)

        assert len(files) == 1
        assert "script.py" in files

    def test_parse_document_skips_template(self):
        """Test parse_document skips template.py files."""
        doc_text = """
================================================================================
File: /path/to/template.py
Relative: template.py
Size: 100 bytes, Lines: 10, Modified: 2024-01-01
================================================================================
def main():
    pass
"""
        files = parse_document(doc_text)

        assert len(files) == 0

    def test_parse_document_skips_non_python(self):
        """Test parse_document skips non-Python files."""
        doc_text = """
================================================================================
File: /path/to/readme.txt
Relative: readme.txt
Size: 100 bytes, Lines: 10, Modified: 2024-01-01
================================================================================
This is a readme file.
"""
        files = parse_document(doc_text)

        assert len(files) == 0

    def test_parse_document_multiple_files(self):
        """Test parse_document with multiple Python files."""
        doc_text = """
================================================================================
File: /path/to/script1.py
Relative: script1.py
Size: 100 bytes, Lines: 10, Modified: 2024-01-01
================================================================================
def main():
    pass

================================================================================
File: /path/to/script2.py
Relative: script2.py
Size: 200 bytes, Lines: 20, Modified: 2024-01-02
================================================================================
def parse_args():
    pass
"""
        files = parse_document(doc_text)

        assert len(files) == 2
        assert "script1.py" in files
        assert "script2.py" in files


class TestGenerateReportV01:
    """Test cases for generate_report function v01."""

    def test_generate_report_empty(self):
        """Test report generation with empty results."""
        results = {}

        # generate_report expects at least one result, skip division by zero
        # with pytest.raises(ZeroDivisionError):
        #     generate_report(results)
        # Actually, let's check if it handles empty gracefully
        # Looking at the source, it will divide by 0 if total_files is 0
        # So we expect a ZeroDivisionError for empty input
        with pytest.raises(ZeroDivisionError):
            generate_report(results)

    def test_generate_report_compliant_file(self):
        """Test report generation with compliant file."""
        file_info = FileInfo(
            relative_path="test.py", content="# test", size=100, lines=5
        )
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
        results = {"test.py": (file_info, compliance)}

        report = generate_report(results)

        assert "COMPLIANT" in report
        assert "test.py" in report

    def test_generate_report_non_compliant_file(self):
        """Test report generation with non-compliant file."""
        file_info = FileInfo(
            relative_path="test.py", content="# test", size=100, lines=5
        )
        compliance = TemplateCompliance()  # All False
        results = {"test.py": (file_info, compliance)}

        report = generate_report(results)

        assert "NON-COMPLIANT" in report
        assert "test.py" in report
        assert "RECOMMENDATIONS" in report

    def test_generate_report_partial_compliance(self):
        """Test report generation with partial compliance."""
        file_info = FileInfo(
            relative_path="test.py", content="# test", size=100, lines=5
        )
        compliance = TemplateCompliance(
            has_main=True,
            has_parse_args=True,
            has_run_main=True,
            has_main_guard=False,  # Missing required
            has_docstring=True,
            has_imports_section=True,
        )
        results = {"test.py": (file_info, compliance)}

        report = generate_report(results)

        assert "PARTIAL" in report
        assert "test.py" in report

    def test_generate_report_component_breakdown(self):
        """Test that report includes component breakdown."""
        file_info = FileInfo(
            relative_path="test.py", content="# test", size=100, lines=5
        )
        compliance = TemplateCompliance(has_main=True)
        results = {"test.py": (file_info, compliance)}

        report = generate_report(results)

        assert "COMPONENT BREAKDOWN" in report
        assert "main() function" in report

    def test_generate_report_missing_components_listed(self):
        """Test that missing required components are listed."""
        file_info = FileInfo(
            relative_path="test.py", content="# test", size=100, lines=5
        )
        compliance = TemplateCompliance(
            has_main=False,
            has_parse_args=False,
            has_run_main=True,
            has_main_guard=True,
        )
        results = {"test.py": (file_info, compliance)}

        report = generate_report(results)

        assert "main()" in report
        assert "parse_args()" in report

    def test_generate_report_all_compliant_message(self):
        """Test that all compliant message appears when all files comply."""
        file_info = FileInfo(
            relative_path="test.py", content="# test", size=100, lines=5
        )
        compliance = TemplateCompliance(
            has_main=True,
            has_parse_args=True,
            has_run_main=True,
            has_main_guard=True,
        )
        results = {"test.py": (file_info, compliance)}

        report = generate_report(results)

        assert "All files are compliant" in report


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
