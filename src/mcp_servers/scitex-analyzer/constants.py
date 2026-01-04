#!/usr/bin/env python3
# Timestamp: "2025-12-28 (ywatanabe)"
# File: ./mcp_servers/scitex-analyzer/constants.py

"""Constants and patterns for SciTeX analyzer."""

# Common SciTeX patterns for detection
SCITEX_PATTERNS = {
    "io_load": r"stx\.io\.load\(['\"]([^'\"]+)['\"]\)",
    "io_save": r"stx\.io\.save\([^,]+,\s*['\"]([^'\"]+)['\"]",
    "io_cache": r"stx\.io\.cache\(['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]",
    "plt_subplots": r"stx\.plt\.subplots\(",
    "set_xyt": r"\.set_xyt\(",
    "config_access": r"CONFIG\.[A-Z_]+\.[A-Z_]+",
    "framework": r"@stx\.main\.run_main",
}

# Anti-patterns to detect
ANTI_PATTERNS = {
    "absolute_path": r"['\"][/\\](home|Users|var|tmp|data)[/\\]",
    "hardcoded_number": r"=\s*(0\.\d+|[1-9]\d*\.?\d*)\s*(?:#|$)",
    "missing_symlink": r"stx\.io\.save\([^)]+\)(?!.*symlink_from_cwd)",
    "mixed_io": r"(pd\.read_|np\.load|plt\.savefig).*stx\.io\.",
}

# Standard library modules for import classification
STDLIB_MODULES = {
    "os",
    "sys",
    "re",
    "json",
    "ast",
    "pathlib",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "typing",
    "math",
    "random",
    "copy",
    "time",
    "subprocess",
    "argparse",
    "logging",
}

# Expected project directories
EXPECTED_DIRECTORIES = ["scripts", "config", "data", "examples", "tests"]

# Standard configuration files
STANDARD_CONFIGS = [
    "config/PATH.yaml",
    "config/PARAMS.yaml",
    "config/COLORS.yaml",
]

# Import order for validation
IMPORT_ORDER = ["stdlib", "third_party", "scitex", "local"]

# Severity priority for sorting violations
SEVERITY_PRIORITY = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# EOF
