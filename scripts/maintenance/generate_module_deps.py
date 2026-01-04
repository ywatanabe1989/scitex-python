#!/usr/bin/env python3
"""
Generate module-level optional dependencies for pyproject.toml.

Scans each scitex submodule for external imports and generates
requirements files and pyproject.toml extras.

Usage:
    python scripts/generate_module_deps.py
    python scripts/generate_module_deps.py --output-dir config/requirements
    python scripts/generate_module_deps.py --format toml
"""

import argparse
import ast
from pathlib import Path

# Standard library modules (Python 3.8+)
STDLIB_MODULES = {
    "abc",
    "aifc",
    "argparse",
    "array",
    "ast",
    "asynchat",
    "asyncio",
    "asyncore",
    "atexit",
    "audioop",
    "base64",
    "bdb",
    "binascii",
    "binhex",
    "bisect",
    "builtins",
    "bz2",
    "calendar",
    "cgi",
    "cgitb",
    "chunk",
    "cmath",
    "cmd",
    "code",
    "codecs",
    "codeop",
    "collections",
    "colorsys",
    "compileall",
    "concurrent",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "cProfile",
    "crypt",
    "csv",
    "ctypes",
    "curses",
    "dataclasses",
    "datetime",
    "dbm",
    "decimal",
    "difflib",
    "dis",
    "distutils",
    "doctest",
    "email",
    "encodings",
    "enum",
    "errno",
    "faulthandler",
    "fcntl",
    "filecmp",
    "fileinput",
    "fnmatch",
    "fractions",
    "ftplib",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "gettext",
    "glob",
    "graphlib",
    "grp",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "idlelib",
    "imaplib",
    "imghdr",
    "imp",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "lib2to3",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "mailbox",
    "mailcap",
    "marshal",
    "math",
    "mimetypes",
    "mmap",
    "modulefinder",
    "multiprocessing",
    "netrc",
    "nis",
    "nntplib",
    "numbers",
    "operator",
    "optparse",
    "os",
    "ossaudiodev",
    "parser",
    "pathlib",
    "pdb",
    "pickle",
    "pickletools",
    "pipes",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posix",
    "posixpath",
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    "resource",
    "rlcompleter",
    "runpy",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtpd",
    "smtplib",
    "sndhdr",
    "socket",
    "socketserver",
    "spwd",
    "sqlite3",
    "ssl",
    "stat",
    "statistics",
    "string",
    "stringprep",
    "struct",
    "subprocess",
    "sunau",
    "symtable",
    "sys",
    "sysconfig",
    "syslog",
    "tabnanny",
    "tarfile",
    "telnetlib",
    "tempfile",
    "termios",
    "test",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tkinter",
    "token",
    "tokenize",
    "trace",
    "traceback",
    "tracemalloc",
    "tty",
    "turtle",
    "turtledemo",
    "types",
    "typing",
    "unicodedata",
    "unittest",
    "urllib",
    "uu",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    "winreg",
    "winsound",
    "wsgiref",
    "xdrlib",
    "xml",
    "xmlrpc",
    "zipapp",
    "zipfile",
    "zipimport",
    "zlib",
    "_thread",
    "__future__",
    "typing_extensions",
}

# Known PyPI packages (import name -> pip package name)
# Only these will be detected as external dependencies
KNOWN_PACKAGES = {
    # Core scientific
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    # ML/AI
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "einops": "einops",
    "optuna": "optuna",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "imblearn": "imbalanced-learn",
    "umap": "umap-learn",
    "faiss": "faiss-cpu",
    # LLM APIs
    "openai": "openai",
    "anthropic": "anthropic",
    "groq": "groq",
    "tiktoken": "tiktoken",
    # Audio/TTS
    "pyttsx3": "pyttsx3",
    "gtts": "gTTS",
    "pydub": "pydub",
    "elevenlabs": "elevenlabs",
    "sounddevice": "sounddevice",
    # Web/Network
    "requests": "requests",
    "aiohttp": "aiohttp",
    "httpx": "httpx",
    "flask": "flask",
    "fastapi": "fastapi",
    "streamlit": "streamlit",
    "websockets": "websockets",
    "mcp": "mcp",
    "fastmcp": "fastmcp",
    # Browser automation
    "selenium": "selenium",
    "playwright": "playwright",
    "bs4": "beautifulsoup4",
    "crawl4ai": "crawl4ai",
    # Data formats
    "h5py": "h5py",
    "zarr": "zarr",
    "xarray": "xarray",
    "openpyxl": "openpyxl",
    "xlrd": "xlrd",
    "yaml": "PyYAML",
    "ruamel": "ruamel.yaml",
    "lxml": "lxml",
    "xmltodict": "xmltodict",
    "bibtexparser": "bibtexparser",
    # Image
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "piexif": "piexif",
    "pypdf": "pypdf",
    "PyPDF2": "PyPDF2",
    "fitz": "PyMuPDF",
    "pdfplumber": "pdfplumber",
    "pytesseract": "pytesseract",
    "qrcode": "qrcode",
    # Document
    "docx": "python-docx",
    "pypandoc": "pypandoc",
    # Database
    "sqlalchemy": "sqlalchemy",
    "psycopg2": "psycopg2-binary",
    # Neuroscience
    "mne": "mne",
    "obspy": "obspy",
    "pyedflib": "pyedflib",
    "pybids": "pybids",
    "tensorpac": "tensorpac",
    # Utilities
    "tqdm": "tqdm",
    "click": "click",
    "rich": "rich",
    "tabulate": "tabulate",
    "natsort": "natsort",
    "joblib": "joblib",
    "psutil": "psutil",
    "packaging": "packaging",
    "pydantic": "pydantic",
    "watchdog": "watchdog",
    "tenacity": "tenacity",
    "chardet": "chardet",
    "thefuzz": "thefuzz",
    "pyperclip": "pyperclip",
    "icecream": "icecream",
    # Git
    "git": "GitPython",
    # Jupyter
    "IPython": "ipython",
    "ipykernel": "ipykernel",
    "jupyterlab": "jupyterlab",
    # Embeddings
    "sentence_transformers": "sentence-transformers",
    # Scholar-specific
    "scholarly": "scholarly",
    "pymed": "pymed",
    "feedparser": "feedparser",
    "pyzotero": "pyzotero",
    "impact_factor": "impact-factor",
    # GUI
    "dearpygui": "dearpygui",
    "PyQt6": "PyQt6",
    "PyQt5": "PyQt5",
    "pyautogui": "pyautogui",
    # Stats
    "statsmodels": "statsmodels",
    "sympy": "sympy",
    # DSP
    "julius": "julius",
    # Misc
    "nest_asyncio": "nest-asyncio",
    "reportlab": "reportlab",
    "magic": "python-magic",
    "dotenv": "python-dotenv",
}

# Modules that are part of scitex (skip these)
INTERNAL_MODULES = {"scitex", "mngs"}


def extract_imports(filepath: Path) -> set:
    """Extract all import statements from a Python file."""
    imports = set()

    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return imports

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                imports.add(module)
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports
            if node.level > 0:
                continue
            if node.module:
                module = node.module.split(".")[0]
                imports.add(module)

    return imports


def get_package_name(import_name: str) -> str | None:
    """Convert import name to pip package name."""
    if import_name in STDLIB_MODULES:
        return None
    if import_name in INTERNAL_MODULES:
        return None
    if import_name in KNOWN_PACKAGES:
        return KNOWN_PACKAGES[import_name]
    return None  # Only return known packages


def scan_module(module_path: Path) -> set:
    """Scan a module directory for all external dependencies."""
    deps = set()

    for py_file in module_path.rglob("*.py"):
        # Skip test files and __pycache__
        if "__pycache__" in str(py_file) or "test_" in py_file.name:
            continue

        imports = extract_imports(py_file)
        for imp in imports:
            pkg = get_package_name(imp)
            if pkg:
                deps.add(pkg)

    return deps


def generate_requirements(src_dir: Path, output_dir: Path | None = None):
    """Generate requirements files for each module."""
    modules_dir = src_dir / "scitex"
    module_deps = {}

    for item in sorted(modules_dir.iterdir()):
        if item.is_dir() and not item.name.startswith((".", "_")):
            if item.name == "__pycache__":
                continue
            deps = scan_module(item)
            if deps:
                module_deps[item.name] = sorted(deps)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for module, deps in module_deps.items():
            req_file = output_dir / f"{module}.txt"
            with open(req_file, "w") as f:
                f.write(f"# Dependencies for scitex.{module}\n")
                f.write("# Auto-generated by scripts/generate_module_deps.py\n\n")
                for dep in deps:
                    f.write(f"{dep}\n")
            print(f"Generated: {req_file}")

    return module_deps


def generate_toml_extras(module_deps: dict) -> str:
    """Generate TOML format optional-dependencies section."""
    lines = [
        "# ============================================",
        "# Module-Level Optional Dependencies",
        "# ============================================",
        "# Install: pip install scitex[module_name]",
        "# Multiple: pip install scitex[audio,scholar,plt]",
        "# All: pip install scitex[all]",
        "",
        "[project.optional-dependencies]",
        "",
    ]

    for module, deps in sorted(module_deps.items()):
        lines.append(f"# scitex.{module}")
        lines.append(f"{module} = [")
        for dep in deps:
            lines.append(f'    "{dep}",')
        lines.append("]")
        lines.append("")

    # Generate 'all' extra
    all_deps = set()
    for deps in module_deps.values():
        all_deps.update(deps)

    lines.append("# All module dependencies")
    lines.append("all = [")
    for dep in sorted(all_deps):
        lines.append(f'    "{dep}",')
    lines.append("]")
    lines.append("")

    # Dev dependencies (separate)
    lines.extend(
        [
            "# Development tools",
            "dev = [",
            '    "pytest",',
            '    "pytest-cov",',
            '    "pytest-xdist",',
            '    "pytest-timeout",',
            '    "pytest-mock",',
            '    "pytest-asyncio",',
            '    "ruff",',
            '    "mypy",',
            '    "pre-commit",',
            "]",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate module-level dependencies for scitex"
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path(__file__).parent.parent / "src",
        help="Source directory containing scitex package",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for requirements files",
    )
    parser.add_argument(
        "--format",
        choices=["requirements", "toml", "both"],
        default="both",
        help="Output format",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.src_dir.parent / "config" / "requirements"

    print(f"Scanning: {args.src_dir}")

    if args.format in ("requirements", "both"):
        module_deps = generate_requirements(args.src_dir, args.output_dir)
    else:
        module_deps = generate_requirements(args.src_dir, None)

    if args.format in ("toml", "both"):
        toml_output = generate_toml_extras(module_deps)
        toml_file = args.output_dir / "extras.toml"
        with open(toml_file, "w") as f:
            f.write(toml_output)
        print(f"\nGenerated TOML extras: {toml_file}")
        print("\n" + "=" * 60)
        print("Copy the following to pyproject.toml:")
        print("=" * 60)
        print(toml_output)


if __name__ == "__main__":
    main()
