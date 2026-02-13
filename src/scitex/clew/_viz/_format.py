#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/_viz/_format.py
"""Formatting functions for verification output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .._chain import (
    ChainVerification,
    RunVerification,
    VerificationStatus,
    verify_run,
)
from ._colors import Colors, status_icon, status_text


def format_run_verification(
    verification: RunVerification,
    verbose: bool = False,
) -> str:
    """
    Format run verification result as a string.

    Parameters
    ----------
    verification : RunVerification
        Verification result
    verbose : bool, optional
        Show detailed file information

    Returns
    -------
    str
        Formatted string
    """
    lines = []
    icon = status_icon(verification.status)
    stat_text = status_text(verification.status)

    lines.append(f"{icon} {verification.session_id} [{stat_text}]")

    if verification.script_path:
        lines.append(f"   Script: {verification.script_path}")

    if verbose or not verification.is_verified:
        if verification.inputs:
            lines.append("   Inputs:")
            for f in verification.inputs:
                f_icon = status_icon(f.status)
                lines.append(f"     {f_icon} {f.path}")

        if verification.outputs:
            lines.append("   Outputs:")
            for f in verification.outputs:
                f_icon = status_icon(f.status)
                lines.append(f"     {f_icon} {f.path}")

        if verification.mismatched_files:
            lines.append(f"   {Colors.RED}Mismatched:{Colors.RESET}")
            for f in verification.mismatched_files:
                lines.append(f"     - {f.path}")
                lines.append(f"       Expected: {f.expected_hash[:16]}...")
                if f.current_hash:
                    lines.append(f"       Got:      {f.current_hash[:16]}...")

        if verification.missing_files:
            lines.append(f"   {Colors.YELLOW}Missing:{Colors.RESET}")
            for f in verification.missing_files:
                lines.append(f"     - {f.path}")

    return "\n".join(lines)


def format_run_detailed(verification: RunVerification) -> str:
    """
    Format run verification with detailed breakdown.

    Shows inputs/scripts/outputs with individual status icons.

    Parameters
    ----------
    verification : RunVerification
        Verification result

    Returns
    -------
    str
        Formatted string with tree structure
    """
    lines = []
    icon = status_icon(verification.status)

    lines.append(f"{icon} {verification.session_id}")

    if verification.script_path:
        script_name = Path(verification.script_path).name
        lines.append(f"   Script: {script_name}")

    failed_inputs = [
        f for f in verification.inputs if f.status != VerificationStatus.VERIFIED
    ]
    failed_outputs = [
        f for f in verification.outputs if f.status != VerificationStatus.VERIFIED
    ]

    if verification.inputs:
        input_icons = "".join([status_icon(f.status) for f in verification.inputs])
        lines.append(f"   ├── inputs:  {input_icons}")
        for f in failed_inputs:
            lines.append(f"   │     └── {Colors.RED}{Path(f.path).name}{Colors.RESET}")

    if verification.script_path:
        script_status = VerificationStatus.VERIFIED
        lines.append(f"   ├── script:  {status_icon(script_status)}")

    if verification.outputs:
        output_icons = "".join([status_icon(f.status) for f in verification.outputs])
        lines.append(f"   └── outputs: {output_icons}")
        for f in failed_outputs:
            lines.append(f"         └── {Colors.RED}{Path(f.path).name}{Colors.RESET}")

    return "\n".join(lines)


def format_chain_verification(
    chain: ChainVerification,
    verbose: bool = False,
) -> str:
    """
    Format chain verification result as a tree.

    Parameters
    ----------
    chain : ChainVerification
        Chain verification result
    verbose : bool, optional
        Show detailed information

    Returns
    -------
    str
        Formatted tree string
    """
    lines = []
    icon = status_icon(chain.status)

    lines.append(
        f"{Colors.BOLD}Chain verification for:{Colors.RESET} {chain.target_file}"
    )
    lines.append(f"Status: {icon} {status_text(chain.status)}")
    lines.append("")

    if not chain.runs:
        lines.append("  (no runs found)")
        return "\n".join(lines)

    for i, run in enumerate(chain.runs):
        is_last = i == len(chain.runs) - 1
        prefix = "└── " if is_last else "├── "
        continuation = "    " if is_last else "│   "

        run_icon = status_icon(run.status)
        lines.append(f"{prefix}{run_icon} {run.session_id}")

        if run.script_path:
            script_name = Path(run.script_path).name
            lines.append(f"{continuation}Script: {script_name}")

        if not run.is_verified:
            if run.mismatched_files:
                lines.append(
                    f"{continuation}{Colors.RED}Mismatched files:{Colors.RESET}"
                )
                for f in run.mismatched_files:
                    lines.append(f"{continuation}  - {Path(f.path).name}")

            if run.missing_files:
                lines.append(
                    f"{continuation}{Colors.YELLOW}Missing files:{Colors.RESET}"
                )
                for f in run.missing_files:
                    lines.append(f"{continuation}  - {Path(f.path).name}")

    return "\n".join(lines)


def format_status(status: Dict[str, Any]) -> str:
    """
    Format verification status summary (like git status).

    Parameters
    ----------
    status : dict
        Status dictionary from get_status()

    Returns
    -------
    str
        Formatted status string
    """
    lines = []

    lines.append(f"{Colors.BOLD}Verification Status{Colors.RESET}")
    lines.append("=" * 40)
    lines.append("")

    total = (
        status["verified_count"] + status["mismatch_count"] + status["missing_count"]
    )
    lines.append(f"Total runs tracked: {total}")
    lines.append(
        f"  {Colors.GREEN}●{Colors.RESET} Verified: {status['verified_count']}"
    )
    lines.append(f"  {Colors.RED}●{Colors.RESET} Mismatch: {status['mismatch_count']}")
    lines.append(
        f"  {Colors.YELLOW}○{Colors.RESET} Missing:  {status['missing_count']}"
    )
    lines.append("")

    if status["mismatched"]:
        lines.append(f"{Colors.RED}Modified (hash mismatch):{Colors.RESET}")
        for item in status["mismatched"][:10]:
            lines.append(f"  {item['session_id']}")
            for f in item["files"][:3]:
                lines.append(f"    └── {Path(f).name}")
            if len(item["files"]) > 3:
                lines.append(f"    └── ... and {len(item['files']) - 3} more")
        if len(status["mismatched"]) > 10:
            lines.append(f"  ... and {len(status['mismatched']) - 10} more runs")
        lines.append("")

    if status["missing"]:
        lines.append(f"{Colors.YELLOW}Missing files:{Colors.RESET}")
        for item in status["missing"][:10]:
            lines.append(f"  {item['session_id']}")
            for f in item["files"][:3]:
                lines.append(f"    └── {Path(f).name}")
            if len(item["files"]) > 3:
                lines.append(f"    └── ... and {len(item['files']) - 3} more")
        if len(status["missing"]) > 10:
            lines.append(f"  ... and {len(status['missing']) - 10} more runs")

    if not status["mismatched"] and not status["missing"]:
        lines.append(f"{Colors.GREEN}All tracked files verified!{Colors.RESET}")

    return "\n".join(lines)


def format_list(
    runs: List[Dict[str, Any]],
    verify: bool = True,
) -> str:
    """
    Format list of runs with verification status.

    Parameters
    ----------
    runs : list of dict
        List of run records from database
    verify : bool, optional
        Whether to verify each run (default: True)

    Returns
    -------
    str
        Formatted list string
    """
    lines = []

    header = f"{'SESSION':<45} {'STATUS':<15} {'SCRIPT':<30}"
    lines.append(f"{Colors.BOLD}{header}{Colors.RESET}")
    lines.append("-" * 90)

    for run in runs:
        session_id = run["session_id"]

        if verify:
            verification = verify_run(session_id)
            icon = status_icon(verification.status)
            stat_text = verification.status.value
        else:
            icon = " "
            stat_text = run.get("status", "unknown")

        script = (
            Path(run.get("script_path", "")).name if run.get("script_path") else "-"
        )

        session_display = session_id[:43] + ".." if len(session_id) > 45 else session_id
        script_display = script[:28] + ".." if len(script) > 30 else script

        lines.append(
            f"{icon} {session_display:<43} {stat_text:<15} {script_display:<30}"
        )

    return "\n".join(lines)


# EOF
