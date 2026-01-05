#!/usr/bin/env python3
# Time-stamp: "2025-06-05 12:00:00 (ywatanabe)"
# File: ./src/scitex/str/_latex.py

"""
LaTeX formatting functions for string manipulation.

Functionality:
    - Convert strings/numbers to LaTeX math mode format
    - Add LaTeX hat notation
Input:
    Strings or numbers to format
Output:
    LaTeX-formatted strings (wrapped in $...$)
Prerequisites:
    None (pure string formatting)
"""


def to_latex_style(str_or_num):
    """
    Convert string or number to LaTeX math mode format.

    Parameters
    ----------
    str_or_num : str or numeric
        Input to format in LaTeX style

    Returns
    -------
    str
        LaTeX-formatted string wrapped in $...$

    Examples
    --------
    >>> to_latex_style('aaa')
    '$aaa$'

    >>> to_latex_style('x^2')
    '$x^2$'

    >>> to_latex_style(123)
    '$123$'
    """
    if not str_or_num and str_or_num != 0:
        return ""

    string = str(str_or_num)

    # Avoid double-wrapping
    if len(string) >= 2 and string[0] == "$" and string[-1] == "$":
        return string
    else:
        return f"${string}$"


def add_hat_in_latex_style(str_or_num):
    """
    Add LaTeX hat notation to string.

    Parameters
    ----------
    str_or_num : str or numeric
        Input to format with hat notation

    Returns
    -------
    str
        LaTeX-formatted string with hat notation

    Examples
    --------
    >>> add_hat_in_latex_style('aaa')
    '$\\hat{aaa}$'

    >>> add_hat_in_latex_style('x')
    '$\\hat{x}$'

    >>> add_hat_in_latex_style(1)
    '$\\hat{1}$'
    """
    if not str_or_num and str_or_num != 0:
        return ""

    hat_latex = rf"\hat{{{str_or_num}}}"
    return f"${hat_latex}$"


# Backward compatibility aliases
latex_style = to_latex_style
hat_latex_style = add_hat_in_latex_style

# Safe versions that are identical (no fallback needed for pure formatting)
safe_to_latex_style = to_latex_style
safe_add_hat_in_latex_style = add_hat_in_latex_style

# EOF
