#!/usr/bin/env python3
# Time-stamp: "2026-01-04 21:00:00 (ywatanabe)"
# File: ./tests/scitex/utils/test__verify_scitex_format.py

"""
Functionality:
    * Tests SciTeX template format verification functionality
    * Validates compliance checking for Python files
    * Tests report generation and file scanning
Input:
    * Test Python file content with various compliance levels
Output:
    * Test results
Prerequisites:
    * pytest
"""

import tempfile
from pathlib import Path

import pytest

from scitex.utils._verify_scitex_format import (
    FileInfo,
    TemplateCompliance,
    _check_run_main_unchanged,
    check_compliance,
    generate_report,
    scan_python_files,
)


class TestTemplateCompliance:
    """Test cases for TemplateCompliance dataclass."""

    def test_compliance_default_values(self):
        """Test default values of TemplateCompliance."""
        compliance = TemplateCompliance()

        assert compliance.has_main is False
        assert compliance.has_parse_args is False
        assert compliance.has_run_main is False
        assert compliance.has_main_guard is False
        assert compliance.is_run_main_unchanged is False
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
            is_run_main_unchanged=True,
        )

        assert compliance.is_compliant is True

    def test_is_compliant_missing_required(self):
        """Test is_compliant property when required items are missing."""
        compliance = TemplateCompliance(
            has_main=True,
            has_parse_args=True,
            has_run_main=True,
            has_main_guard=False,  # Missing
            is_run_main_unchanged=True,
        )

        assert compliance.is_compliant is False

    def test_compliance_score_all_true(self):
        """Test compliance_score when all items are True."""
        compliance = TemplateCompliance(
            has_main=True,
            has_parse_args=True,
            has_run_main=True,
            has_main_guard=True,
            is_run_main_unchanged=True,
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
            is_run_main_unchanged=True,
        )
        # 5 out of 10 items are True
        assert compliance.compliance_score == 0.5


class TestFileInfo:
    """Test cases for FileInfo dataclass."""

    def test_file_info_creation(self):
        """Test FileInfo creation."""
        info = FileInfo(
            relative_path="test/file.py", content="print('hello')", size=100, lines=5
        )

        assert info.relative_path == "test/file.py"
        assert info.content == "print('hello')"
        assert info.size == 100
        assert info.lines == 5


class TestCheckCompliance:
    """Test cases for check_compliance function."""

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

    def test_check_compliance_full_template(self):
        """Test check_compliance with a fully compliant template."""
        content = '''#!/usr/bin/env python3
"""
Module docstring
"""

"""Imports"""
import argparse
import scitex as stx

"""Functions & Classes"""

def main(args):
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

def run_main() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, rng

    import sys
    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(
        sys, plt, verbose=False
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=False,
    )

if __name__ == "__main__":
    run_main()
'''
        compliance = check_compliance(content)

        assert compliance.has_main is True
        assert compliance.has_parse_args is True
        assert compliance.has_run_main is True
        assert compliance.has_main_guard is True
        assert compliance.has_docstring is True
        assert compliance.has_imports_section is True
        assert compliance.has_functions_classes_section is True
        assert compliance.uses_scitex_session is True
        assert compliance.has_verbose_param is True


class TestCheckRunMainUnchanged:
    """Test cases for _check_run_main_unchanged function."""

    def test_run_main_template_format(self):
        """Test _check_run_main_unchanged with correct template format."""
        content = '''
def run_main() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, rng

    import sys
    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(
        sys, plt
    )

    exit_status = main(args)

    stx.session.close(CONFIG)
'''
        assert _check_run_main_unchanged(content) is True

    def test_run_main_missing_docstring(self):
        """Test _check_run_main_unchanged with missing docstring."""
        content = """
def run_main() -> None:
    global CONFIG, CC, sys, plt, rng
    import sys
"""
        assert _check_run_main_unchanged(content) is False

    def test_run_main_missing_global(self):
        """Test _check_run_main_unchanged with missing global declaration."""
        content = '''
def run_main() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    import sys
'''
        assert _check_run_main_unchanged(content) is False


class TestScanPythonFiles:
    """Test cases for scan_python_files function."""

    def test_scan_single_file(self):
        """Test scanning a single Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / "test_script.py"
            test_file.write_text("print('hello')")

            files = scan_python_files([test_file], tmppath)

            assert len(files) == 1
            assert "test_script.py" in files
            assert files["test_script.py"].content == "print('hello')"

    def test_scan_directory(self):
        """Test scanning a directory for Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "script1.py").write_text("# script 1")
            (tmppath / "script2.py").write_text("# script 2")
            (tmppath / "not_python.txt").write_text("not python")

            files = scan_python_files([tmppath], tmppath)

            assert len(files) == 2
            assert "script1.py" in files
            assert "script2.py" in files

    def test_scan_skips_template(self):
        """Test that template.py is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "template.py").write_text("# template")
            (tmppath / "script.py").write_text("# script")

            files = scan_python_files([tmppath], tmppath)

            assert len(files) == 1
            assert "script.py" in files
            assert "template.py" not in files

    def test_scan_skips_init(self):
        """Test that __init__.py is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "__init__.py").write_text("# init")
            (tmppath / "script.py").write_text("# script")

            files = scan_python_files([tmppath], tmppath)

            assert len(files) == 1
            assert "script.py" in files
            assert "__init__.py" not in files

    def test_scan_nonexistent_path(self):
        """Test scanning non-existent path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            nonexistent = tmppath / "nonexistent"

            files = scan_python_files([nonexistent], tmppath)

            assert len(files) == 0


class TestGenerateReport:
    """Test cases for generate_report function."""

    def test_generate_report_empty(self):
        """Test report generation with empty results."""
        results = {}

        report = generate_report(results)

        assert "TEMPLATE COMPLIANCE REPORT" in report
        assert "Total Python files analyzed: 0" in report

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
            is_run_main_unchanged=True,
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
        """Test report generation with partial compliance.

        PARTIAL status requires:
        - is_compliant = False (missing a required item)
        - compliance_score >= 0.5 (at least 5/10 items True)
        """
        file_info = FileInfo(
            relative_path="test.py", content="# test", size=100, lines=5
        )
        # 6 out of 10 items True (60% score) but is_run_main_unchanged=False
        # makes is_compliant=False, resulting in PARTIAL status
        compliance = TemplateCompliance(
            has_main=True,
            has_parse_args=True,
            has_run_main=True,
            has_main_guard=True,
            is_run_main_unchanged=False,  # Missing required
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


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
