#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/cli/scholar/_utils.py
# ----------------------------------------

"""Shared utilities for Scholar CLI commands."""

from __future__ import annotations

import json

import click


def output_json(data: dict) -> None:
    """Output data as JSON."""
    click.echo(json.dumps(data, indent=2, default=str))


def output_result(data: dict, json_mode: bool) -> None:
    """Output result in appropriate format."""
    from scitex import logging

    logger = logging.getLogger(__name__)

    if json_mode:
        output_json(data)
    else:
        if data.get("success"):
            logger.success(data.get("message", "Success"))
        else:
            logger.error(data.get("error", "Failed"))


# EOF
