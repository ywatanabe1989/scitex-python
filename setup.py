#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-08-02 19:34:26 (ywatanabe)"
# File: /home/ywatanabe/proj/SciTeX-Code/setup.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./setup.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""Setup script for SciTeX package."""

from setuptools import find_packages, setup

# Read version from __version__.py
version_file = os.path.join("src", "scitex", "__version__.py")
version_dict = {}
with open(version_file) as f:
    exec(f.read(), version_dict)
version = version_dict["__version__"]


def read_requirements(filename):
    """Read requirements from a file."""
    filepath = os.path.join("requirements", filename)
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


# Core dependencies (always installed)
install_requires = read_requirements("core.txt")

# Optional dependency groups
extras_require = {
    # Deep Learning (Heavy - 2-4 GB)
    "dl": read_requirements("dl.txt"),
    
    # AI APIs
    "ai-apis": read_requirements("ai-apis.txt"),
    
    # Scholar module (paper management)
    "scholar": read_requirements("scholar.txt"),
    
    # Additional ML tools
    "ml": read_requirements("ml.txt"),
    
    # Neuroscience & specialized
    "neuro": read_requirements("neuro.txt"),
    
    # Web frameworks
    "web": read_requirements("web.txt"),
    
    # Jupyter notebooks
    "jupyter": read_requirements("jupyter.txt"),
    
    # Extra utilities
    "extras": read_requirements("extras.txt"),
    
    # Development tools
    "dev": read_requirements("dev.txt"),
}

# Convenience groups
extras_require["recommended"] = (
    extras_require["ml"] + 
    extras_require["jupyter"]
)

extras_require["all"] = list(set(
    sum(
        [v for k, v in extras_require.items() if k not in ["dev", "all"]],
        []
    )
))

# Read README for long description
readme_path = "README.md"
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="scitex",
    version=version,
    author="Yusuke Watanabe",
    author_email="ywatanabe@scitex.ai",
    description="A comprehensive scientific computing framework for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ywatanabe1989/scitex-code",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "scitex": ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.txt"],
    },
    entry_points={
        "console_scripts": [
            "scitex=scitex.cli:cli",
        ],
    },
    extras_require=extras_require,
    keywords="scientific computing, data analysis, machine learning, visualization, research",
)

# EOF
