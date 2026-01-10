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
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/utils/_verify_scitex_format_v01.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-09 21:28:40 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/utils/_verify_scitex_format.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/utils/_verify_scitex_format.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# """Analyze template compliance from document text."""
# 
# import re
# from dataclasses import dataclass
# from typing import Dict
# 
# 
# @dataclass
# class FileInfo:
#     """Store file information."""
# 
#     relative_path: str
#     content: str
#     size: int
#     lines: int
# 
# 
# @dataclass
# class TemplateCompliance:
#     """Store compliance check results."""
# 
#     has_main: bool = False
#     has_parse_args: bool = False
#     has_run_main: bool = False
#     has_main_guard: bool = False
#     has_docstring: bool = False
#     has_imports_section: bool = False
#     has_functions_classes_section: bool = False
#     uses_scitex_session: bool = False
#     has_verbose_param: bool = False
# 
#     @property
#     def is_compliant(self) -> bool:
#         """Check if file is fully compliant."""
#         return all(
#             [
#                 self.has_main,
#                 self.has_parse_args,
#                 self.has_run_main,
#                 self.has_main_guard,
#             ]
#         )
# 
#     @property
#     def compliance_score(self) -> float:
#         """Calculate compliance score (0-1)."""
#         checks = [
#             self.has_main,
#             self.has_parse_args,
#             self.has_run_main,
#             self.has_main_guard,
#             self.has_docstring,
#             self.has_imports_section,
#             self.has_functions_classes_section,
#             self.uses_scitex_session,
#             self.has_verbose_param,
#         ]
#         return sum(checks) / len(checks)
# 
# 
# def parse_document(doc_text: str) -> Dict[str, FileInfo]:
#     """Parse the document to extract Python files."""
#     files = {}
# 
#     # Pattern to match file headers and content
#     file_pattern = re.compile(
#         r"={80,}\n"
#         r"File: (.+?)\n"
#         r"Relative: (.+?)\n"
#         r"Size: (\d+) bytes, Lines: (\d+),.*?\n"
#         r"={80,}\n"
#         r"(.*?)"
#         r"(?=\n={80,}\nFile:|$)",
#         re.DOTALL,
#     )
# 
#     for match in file_pattern.finditer(doc_text):
#         full_path = match.group(1)
#         relative_path = match.group(2)
#         size = int(match.group(3))
#         lines = int(match.group(4))
#         content = match.group(5)
# 
#         # Only process .py files, skip template.py
#         if relative_path.endswith(".py") and "template.py" not in relative_path:
#             files[relative_path] = FileInfo(
#                 relative_path=relative_path,
#                 content=content,
#                 size=size,
#                 lines=lines,
#             )
# 
#     return files
# 
# 
# def check_compliance(content: str) -> TemplateCompliance:
#     """Check if content follows template format."""
#     compliance = TemplateCompliance()
# 
#     # Check for main function
#     compliance.has_main = bool(re.search(r"^def main\(", content, re.MULTILINE))
# 
#     # Check for parse_args function
#     compliance.has_parse_args = bool(
#         re.search(r"^def parse_args\(", content, re.MULTILINE)
#     )
# 
#     # Check for run_main function
#     compliance.has_run_main = bool(re.search(r"^def run_main\(", content, re.MULTILINE))
# 
#     # Check for main guard
#     compliance.has_main_guard = bool(re.search(r'if __name__ == "__main__":', content))
# 
#     # Check for module docstring
#     compliance.has_docstring = bool(re.search(r'"""[\s\S]*?"""', content))
# 
#     # Check for section markers
#     compliance.has_imports_section = bool(re.search(r'"""Imports"""', content))
#     compliance.has_functions_classes_section = bool(
#         re.search(r'"""Functions & Classes"""', content)
#     )
# 
#     # Check for scitex session usage
#     compliance.uses_scitex_session = bool(re.search(r"stx\.session\.start", content))
# 
#     # Check for verbose parameter
#     compliance.has_verbose_param = bool(
#         re.search(r"verbose\s*[:=]\s*(?:bool|True|False)", content)
#     )
# 
#     return compliance
# 
# 
# def generate_report(results: Dict[str, tuple]) -> str:
#     """Generate compliance report."""
#     lines = []
#     lines.append("=" * 80)
#     lines.append("TEMPLATE COMPLIANCE REPORT")
#     lines.append("=" * 80)
#     lines.append("")
# 
#     # Summary statistics
#     total_files = len(results)
#     compliant_files = sum(1 for _, c in results.values() if c.is_compliant)
#     avg_score = (
#         sum(c.compliance_score for _, c in results.values()) / total_files
#         if total_files > 0
#         else 0
#     )
# 
#     lines.append(f"Total Python files analyzed: {total_files}")
#     lines.append(
#         f"Fully compliant files: {compliant_files} ({compliant_files / total_files * 100:.1f}%)"
#     )
#     lines.append(f"Average compliance score: {avg_score:.1%}")
#     lines.append("")
# 
#     # Detailed results
#     lines.append("=" * 80)
#     lines.append("DETAILED RESULTS (sorted by compliance score)")
#     lines.append("=" * 80)
#     lines.append("")
# 
#     # Sort by compliance score (lowest first)
#     sorted_results = sorted(
#         results.items(), key=lambda x: (x[1][1].compliance_score, x[0])
#     )
# 
#     for filepath, (file_info, compliance) in sorted_results:
#         score = compliance.compliance_score
# 
#         # Status indicator
#         if compliance.is_compliant:
#             status = "✓ COMPLIANT"
#         elif score >= 0.5:
#             status = "⚠ PARTIAL"
#         else:
#             status = "✗ NON-COMPLIANT"
# 
#         lines.append(f"{status} [{score:.0%}] {filepath}")
#         lines.append(f"  Lines: {file_info.lines}, Size: {file_info.size} bytes")
# 
#         # Show missing components
#         if not compliance.is_compliant:
#             missing = []
#             if not compliance.has_main:
#                 missing.append("main()")
#             if not compliance.has_parse_args:
#                 missing.append("parse_args()")
#             if not compliance.has_run_main:
#                 missing.append("run_main()")
#             if not compliance.has_main_guard:
#                 missing.append("if __name__ == '__main__'")
# 
#             if missing:
#                 lines.append(f"  ❌ REQUIRED Missing: {', '.join(missing)}")
# 
#         # Show optional items
#         optional_missing = []
#         if not compliance.has_docstring:
#             optional_missing.append("module docstring")
#         if not compliance.has_imports_section:
#             optional_missing.append("'Imports' section")
#         if not compliance.has_functions_classes_section:
#             optional_missing.append("'Functions & Classes' section")
#         if not compliance.uses_scitex_session:
#             optional_missing.append("scitex session")
#         if not compliance.has_verbose_param:
#             optional_missing.append("verbose parameter")
# 
#         if optional_missing:
#             lines.append(f"  ⚠️  OPTIONAL Missing: {', '.join(optional_missing)}")
# 
#         lines.append("")
# 
#     # Component breakdown
#     lines.append("=" * 80)
#     lines.append("COMPONENT BREAKDOWN")
#     lines.append("=" * 80)
#     lines.append("")
# 
#     components = [
#         (
#             "✓ main() function",
#             sum(1 for _, c in results.values() if c.has_main),
#         ),
#         (
#             "✓ parse_args() function",
#             sum(1 for _, c in results.values() if c.has_parse_args),
#         ),
#         (
#             "✓ run_main() function",
#             sum(1 for _, c in results.values() if c.has_run_main),
#         ),
#         (
#             "✓ if __name__ == '__main__'",
#             sum(1 for _, c in results.values() if c.has_main_guard),
#         ),
#         (
#             "  Module docstring",
#             sum(1 for _, c in results.values() if c.has_docstring),
#         ),
#         (
#             "  'Imports' section",
#             sum(1 for _, c in results.values() if c.has_imports_section),
#         ),
#         (
#             "  'Functions & Classes'",
#             sum(1 for _, c in results.values() if c.has_functions_classes_section),
#         ),
#         (
#             "  scitex session",
#             sum(1 for _, c in results.values() if c.uses_scitex_session),
#         ),
#         (
#             "  verbose parameter",
#             sum(1 for _, c in results.values() if c.has_verbose_param),
#         ),
#     ]
# 
#     for component, count in components:
#         percentage = count / total_files * 100 if total_files > 0 else 0
#         marker = "REQUIRED" if component.startswith("✓") else "OPTIONAL"
#         lines.append(
#             f"[{marker}] {component:.<45} {count}/{total_files} ({percentage:.1f}%)"
#         )
# 
#     # Recommendations
#     lines.append("")
#     lines.append("=" * 80)
#     lines.append("RECOMMENDATIONS")
#     lines.append("=" * 80)
#     lines.append("")
# 
#     non_compliant = [
#         (path, compliance)
#         for path, (_, compliance) in results.items()
#         if not compliance.is_compliant
#     ]
# 
#     if non_compliant:
#         lines.append(
#             f"Found {len(non_compliant)} non-compliant files that need updates:"
#         )
#         lines.append("")
#         for path, compliance in non_compliant:
#             lines.append(f"• {path}")
#             fixes = []
#             if not compliance.has_main:
#                 fixes.append("  - Add main(args) function")
#             if not compliance.has_parse_args:
#                 fixes.append("  - Add parse_args() function")
#             if not compliance.has_run_main:
#                 fixes.append("  - Add run_main() function")
#             if not compliance.has_main_guard:
#                 fixes.append("  - Add if __name__ == '__main__': run_main()")
#             for fix in fixes:
#                 lines.append(fix)
#             lines.append("")
#     else:
#         lines.append("✓ All files are compliant! Great job!")
# 
#     return "\n".join(lines)
# 
# 
# def main():
#     """Main function."""
#     # Read the document
#     with open("/mnt/user-data/uploads/document_1.txt", "r", encoding="utf-8") as f:
#         doc_text = f.read()
# 
#     print("Parsing document...")
#     files = parse_document(doc_text)
#     print(f"Found {len(files)} Python files")
#     print()
# 
#     # Check compliance for each file
#     results = {}
#     for path, file_info in files.items():
#         compliance = check_compliance(file_info.content)
#         results[path] = (file_info, compliance)
# 
#     # Generate report
#     report = generate_report(results)
#     print(report)
# 
#     # Save report
#     report_path = "/home/claude/template_compliance_report.txt"
#     with open(report_path, "w") as f:
#         f.write(report)
#     print(f"\nReport saved to: {report_path}")
# 
# 
# if __name__ == "__main__":
#     main()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/utils/_verify_scitex_format_v01.py
# --------------------------------------------------------------------------------
