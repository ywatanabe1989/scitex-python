#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_tables/_latex/_validator.py

"""LaTeX validation for error detection and suggestions."""

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class ErrorSeverity(Enum):
    """Error severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class LaTeXError:
    """A LaTeX validation error.

    Attributes:
        line: Line number (1-indexed)
        column: Column number (1-indexed)
        message: Error description
        severity: Error severity
        code: Error code (e.g., E001, W001)
        suggestion: Fix suggestion
        context: Line content for context
    """

    line: int
    column: int
    message: str
    severity: ErrorSeverity
    code: str
    suggestion: Optional[str] = None
    context: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of LaTeX validation.

    Attributes:
        is_valid: Whether the LaTeX is valid (no errors)
        errors: List of errors
        warnings: List of warnings
        info: List of info messages
    """

    is_valid: bool = True
    errors: List[LaTeXError] = field(default_factory=list)
    warnings: List[LaTeXError] = field(default_factory=list)
    info: List[LaTeXError] = field(default_factory=list)

    def add_error(self, error: LaTeXError) -> None:
        """Add an error to the appropriate list."""
        if error.severity == ErrorSeverity.ERROR:
            self.errors.append(error)
            self.is_valid = False
        elif error.severity == ErrorSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.info.append(error)

    @property
    def all_issues(self) -> List[LaTeXError]:
        """Get all issues sorted by line number."""
        return sorted(
            self.errors + self.warnings + self.info,
            key=lambda e: (e.line, e.column),
        )


def validate_latex(
    latex_code: str,
    level: str = "syntax",
) -> ValidationResult:
    """Validate LaTeX code.

    Args:
        latex_code: LaTeX source code
        level: Validation level ("syntax", "semantic", "compile")

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult()
    lines = latex_code.split("\n")

    # Always do syntax validation
    _validate_syntax(lines, result)

    # Semantic validation
    if level in ("semantic", "compile"):
        _validate_semantic(lines, result)

    # Compilation validation (optional, requires pdflatex)
    if level == "compile":
        _validate_compilation(latex_code, result)

    return result


def _validate_syntax(lines: List[str], result: ValidationResult) -> None:
    """Check for syntax errors.

    Args:
        lines: Source lines
        result: ValidationResult to populate
    """
    # Track brace balance
    brace_stack = []  # List of (line, col, char)
    env_stack = []  # List of (line, env_name)

    for line_num, line in enumerate(lines, 1):
        # Skip comments
        line_no_comment = line.split("%")[0] if "%" in line and "\\%" not in line.split("%")[0] else line

        # Check brace balance
        for col, char in enumerate(line_no_comment, 1):
            if char == "{":
                # Check if escaped
                if col > 1 and line_no_comment[col - 2] == "\\":
                    continue
                brace_stack.append((line_num, col, char))
            elif char == "}":
                if col > 1 and line_no_comment[col - 2] == "\\":
                    continue
                if brace_stack:
                    brace_stack.pop()
                else:
                    result.add_error(
                        LaTeXError(
                            line=line_num,
                            column=col,
                            message="Unmatched closing brace '}'",
                            severity=ErrorSeverity.ERROR,
                            code="E001",
                            suggestion="Remove the extra '}' or add a matching '{'",
                            context=line,
                        )
                    )

        # Check environment matching
        begin_match = re.search(r"\\begin\{(\w+)\}", line_no_comment)
        if begin_match:
            env_name = begin_match.group(1)
            env_stack.append((line_num, env_name))

        end_match = re.search(r"\\end\{(\w+)\}", line_no_comment)
        if end_match:
            env_name = end_match.group(1)
            if env_stack:
                _, expected_env = env_stack[-1]
                if env_name == expected_env:
                    env_stack.pop()
                else:
                    result.add_error(
                        LaTeXError(
                            line=line_num,
                            column=end_match.start() + 1,
                            message=f"Environment mismatch: expected \\end{{{expected_env}}}, got \\end{{{env_name}}}",
                            severity=ErrorSeverity.ERROR,
                            code="E002",
                            suggestion=f"Change to \\end{{{expected_env}}} or fix the \\begin command",
                            context=line,
                        )
                    )
            else:
                result.add_error(
                    LaTeXError(
                        line=line_num,
                        column=end_match.start() + 1,
                        message=f"\\end{{{env_name}}} without matching \\begin{{{env_name}}}",
                        severity=ErrorSeverity.ERROR,
                        code="E003",
                        suggestion=f"Add \\begin{{{env_name}}} before this or remove the \\end command",
                        context=line,
                    )
                )

        # Check for common typos
        _check_common_typos(line_num, line, result)

    # Report unclosed braces
    for line_num, col, _ in brace_stack:
        result.add_error(
            LaTeXError(
                line=line_num,
                column=col,
                message="Unclosed brace '{'",
                severity=ErrorSeverity.ERROR,
                code="E004",
                suggestion="Add a matching '}'",
                context=lines[line_num - 1] if line_num <= len(lines) else "",
            )
        )

    # Report unclosed environments
    for line_num, env_name in env_stack:
        result.add_error(
            LaTeXError(
                line=line_num,
                column=1,
                message=f"Unclosed environment: \\begin{{{env_name}}} without \\end{{{env_name}}}",
                severity=ErrorSeverity.ERROR,
                code="E005",
                suggestion=f"Add \\end{{{env_name}}} at the end",
                context=lines[line_num - 1] if line_num <= len(lines) else "",
            )
        )


def _check_common_typos(line_num: int, line: str, result: ValidationResult) -> None:
    """Check for common LaTeX typos.

    Args:
        line_num: Line number
        line: Line content
        result: ValidationResult to populate
    """
    typos = [
        (r"\\beign\{", "\\begin{", "W001"),
        (r"\\beigin\{", "\\begin{", "W001"),
        (r"\\ened\{", "\\end{", "W002"),
        (r"\\ceneter", "\\centering", "W003"),
        (r"\\includegraphcis", "\\includegraphics", "W004"),
        (r"\\captoin", "\\caption", "W005"),
        (r"\\lable\{", "\\label{", "W006"),
        (r"\\toperule", "\\toprule", "W007"),
        (r"\\bottomrul", "\\bottomrule", "W008"),
        (r"\\midruel", "\\midrule", "W009"),
    ]

    for pattern, correct, code in typos:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            result.add_error(
                LaTeXError(
                    line=line_num,
                    column=match.start() + 1,
                    message=f"Possible typo: '{match.group()}' should be '{correct}'",
                    severity=ErrorSeverity.WARNING,
                    code=code,
                    suggestion=f"Replace with '{correct}'",
                    context=line,
                )
            )


def _validate_semantic(lines: List[str], result: ValidationResult) -> None:
    """Check for semantic issues.

    Args:
        lines: Source lines
        result: ValidationResult to populate
    """
    labels = set()
    refs = set()
    has_caption = False
    in_float = False

    for line_num, line in enumerate(lines, 1):
        # Track float environments
        if re.search(r"\\begin\{(figure|table)\}", line):
            in_float = True
            has_caption = False

        if re.search(r"\\end\{(figure|table)\}", line):
            if in_float and not has_caption:
                result.add_error(
                    LaTeXError(
                        line=line_num,
                        column=1,
                        message="Float environment without caption",
                        severity=ErrorSeverity.WARNING,
                        code="W010",
                        suggestion="Add \\caption{...} to the figure/table",
                        context=line,
                    )
                )
            in_float = False

        # Track captions
        if "\\caption" in line:
            has_caption = True

        # Collect labels
        label_match = re.search(r"\\label\{([^}]+)\}", line)
        if label_match:
            label = label_match.group(1)
            if label in labels:
                result.add_error(
                    LaTeXError(
                        line=line_num,
                        column=label_match.start() + 1,
                        message=f"Duplicate label: '{label}'",
                        severity=ErrorSeverity.WARNING,
                        code="W011",
                        suggestion="Use a unique label",
                        context=line,
                    )
                )
            labels.add(label)

        # Collect refs
        for ref_match in re.finditer(r"\\ref\{([^}]+)\}", line):
            refs.add(ref_match.group(1))

    # Check for undefined refs (only warning since refs might be in other files)
    # for ref in refs - labels:
    #     result.add_error(...)


def _validate_compilation(latex_code: str, result: ValidationResult) -> None:
    """Attempt to compile LaTeX and capture errors.

    Args:
        latex_code: LaTeX source code
        result: ValidationResult to populate
    """
    # Create a minimal document if needed
    if "\\documentclass" not in latex_code:
        latex_code = _wrap_in_document(latex_code)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "test.tex"
            tex_path.write_text(latex_code, encoding="utf-8")

            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(tex_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if proc.returncode != 0:
                # Parse pdflatex output for errors
                _parse_pdflatex_output(proc.stdout, result)

    except FileNotFoundError:
        result.add_error(
            LaTeXError(
                line=0,
                column=0,
                message="pdflatex not found - compilation check skipped",
                severity=ErrorSeverity.INFO,
                code="I001",
                suggestion="Install TeX Live or MiKTeX for compilation validation",
            )
        )
    except subprocess.TimeoutExpired:
        result.add_error(
            LaTeXError(
                line=0,
                column=0,
                message="Compilation timed out",
                severity=ErrorSeverity.WARNING,
                code="W012",
                suggestion="Check for infinite loops in macros",
            )
        )
    except Exception as e:
        result.add_error(
            LaTeXError(
                line=0,
                column=0,
                message=f"Compilation error: {str(e)}",
                severity=ErrorSeverity.INFO,
                code="I002",
            )
        )


def _wrap_in_document(content: str) -> str:
    """Wrap content in a minimal LaTeX document.

    Args:
        content: LaTeX content

    Returns:
        Complete document string
    """
    return f"""\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\begin{{document}}
{content}
\\end{{document}}
"""


def _parse_pdflatex_output(output: str, result: ValidationResult) -> None:
    """Parse pdflatex output for errors.

    Args:
        output: pdflatex stdout
        result: ValidationResult to populate
    """
    # Look for error lines
    error_pattern = re.compile(r"^! (.+)$", re.MULTILINE)
    line_pattern = re.compile(r"^l\.(\d+) (.*)$", re.MULTILINE)

    for match in error_pattern.finditer(output):
        error_msg = match.group(1)

        # Try to find line number
        line_num = 0
        context = ""
        line_match = line_pattern.search(output, match.end())
        if line_match:
            line_num = int(line_match.group(1))
            context = line_match.group(2)

        result.add_error(
            LaTeXError(
                line=line_num,
                column=1,
                message=error_msg,
                severity=ErrorSeverity.ERROR,
                code="C001",
                context=context,
            )
        )


__all__ = [
    "ErrorSeverity",
    "LaTeXError",
    "ValidationResult",
    "validate_latex",
]

# EOF
