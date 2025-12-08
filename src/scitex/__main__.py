#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciTeX Package Entry Point

Allows running: python -m scitex [command]
"""

import sys


def main():
    """Main entry point for scitex CLI"""
    try:
        from scitex.cli.main import cli

        cli()
    except ImportError:
        # CLI not available (click not installed)
        print("SciTeX CLI requires 'click' package")
        print("Install: pip install click")
        sys.exit(1)


if __name__ == "__main__":
    main()
