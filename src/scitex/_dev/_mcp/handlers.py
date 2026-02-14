#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/_mcp/handlers.py

"""MCP handler implementations for developer utilities."""

from __future__ import annotations

from typing import Any


async def list_versions_handler(
    packages: list[str] | None = None,
) -> dict[str, Any]:
    """List versions across the scitex ecosystem.

    Parameters
    ----------
    packages : list[str] | None
        List of package names to check. If None, checks all ecosystem packages.

    Returns
    -------
    dict
        Version information for each package.
    """
    from scitex._dev import list_versions

    return list_versions(packages)


async def check_versions_handler(
    packages: list[str] | None = None,
) -> dict[str, Any]:
    """Check version consistency across the scitex ecosystem.

    Parameters
    ----------
    packages : list[str] | None
        List of package names to check. If None, checks all ecosystem packages.

    Returns
    -------
    dict
        Detailed version check results with summary.
    """
    from scitex._dev import check_versions

    return check_versions(packages)


async def check_hosts_handler(
    packages: list[str] | None = None,
    hosts: list[str] | None = None,
) -> dict[str, Any]:
    """Check versions on SSH hosts.

    Parameters
    ----------
    packages : list[str] | None
        List of package names to check. If None, checks all ecosystem packages.
    hosts : list[str] | None
        List of host names to check. If None, checks all enabled hosts.

    Returns
    -------
    dict
        Host name -> package name -> version info mapping.
    """
    from scitex._dev import check_all_hosts

    return check_all_hosts(packages=packages, hosts=hosts)


async def check_remotes_handler(
    packages: list[str] | None = None,
    remotes: list[str] | None = None,
) -> dict[str, Any]:
    """Check versions on GitHub remotes.

    Parameters
    ----------
    packages : list[str] | None
        List of package names to check. If None, checks all ecosystem packages.
    remotes : list[str] | None
        List of remote names to check. If None, checks all enabled remotes.

    Returns
    -------
    dict
        Remote name -> package name -> version info mapping.
    """
    from scitex._dev import check_all_remotes

    return check_all_remotes(packages=packages, remotes=remotes)


async def get_config_handler() -> dict[str, Any]:
    """Get current developer configuration.

    Returns
    -------
    dict
        Configuration including packages, hosts, remotes, branches.
    """
    from scitex._dev import get_config_path, load_config

    config = load_config()
    return {
        "config_path": str(get_config_path()),
        "packages": [
            {
                "name": p.name,
                "local_path": p.local_path,
                "pypi_name": p.pypi_name,
                "github_repo": p.github_repo,
            }
            for p in config.packages
        ],
        "hosts": [
            {
                "name": h.name,
                "hostname": h.hostname,
                "user": h.user,
                "role": h.role,
                "enabled": h.enabled,
            }
            for h in config.hosts
        ],
        "github_remotes": [
            {"name": r.name, "org": r.org, "enabled": r.enabled}
            for r in config.github_remotes
        ],
        "branches": config.branches,
    }


async def get_full_versions_handler(
    packages: list[str] | None = None,
    include_hosts: bool = False,
    include_remotes: bool = False,
) -> dict[str, Any]:
    """Get comprehensive version data from all sources.

    Parameters
    ----------
    packages : list[str] | None
        List of package names to check. If None, checks all ecosystem packages.
    include_hosts : bool
        Include SSH host version checks.
    include_remotes : bool
        Include GitHub remote version checks.

    Returns
    -------
    dict
        Combined version data from local, hosts, and remotes.
    """
    from scitex._dev import check_all_hosts, check_all_remotes, list_versions

    result = {
        "packages": list_versions(packages),
        "hosts": {},
        "remotes": {},
    }

    if include_hosts:
        try:
            result["hosts"] = check_all_hosts(packages=packages)
        except Exception as e:
            result["hosts"] = {"error": str(e)}

    if include_remotes:
        try:
            result["remotes"] = check_all_remotes(packages=packages)
        except Exception as e:
            result["remotes"] = {"error": str(e)}

    return result


async def test_run_handler(
    module: str = "",
    fast: bool = False,
    coverage: bool = False,
    exitfirst: bool = True,
    pattern: str = "",
    parallel: str = "auto",
) -> dict[str, Any]:
    """Run tests locally.

    Parameters
    ----------
    module : str
        Module to test (e.g., "stats", "io"). Empty for all.
    fast : bool
        Skip slow tests.
    coverage : bool
        Enable coverage.
    exitfirst : bool
        Stop on first failure.
    pattern : str
        Test name filter pattern.
    parallel : str
        Parallel workers ("auto", "0", or number).

    Returns
    -------
    dict
        {"exit_code": int}
    """
    from scitex._dev._test import TestConfig, run_local

    config = TestConfig(
        module=module,
        fast=fast,
        coverage=coverage,
        exitfirst=exitfirst,
        pattern=pattern,
        parallel=parallel,
    )
    exit_code = run_local(config)
    return {"exit_code": exit_code}


async def test_hpc_run_handler(
    module: str = "",
    fast: bool = False,
    hpc_cpus: int = 8,
    hpc_partition: str = "sapphire",
    hpc_time: str = "00:10:00",
    hpc_mem: str = "16G",
    async_mode: bool = False,
) -> dict[str, Any]:
    """Run tests on HPC via Slurm.

    Parameters
    ----------
    module : str
        Module to test.
    fast : bool
        Skip slow tests.
    hpc_cpus : int
        CPUs per task.
    hpc_partition : str
        Slurm partition.
    hpc_time : str
        Time limit.
    hpc_mem : str
        Memory limit.
    async_mode : bool
        If True, submit via sbatch (returns job ID).
        If False, run blocking via srun.

    Returns
    -------
    dict
        {"exit_code": int} for srun, {"job_id": str} for sbatch.
    """
    from scitex._dev._test import (
        TestConfig,
        _check_ssh,
        _get_hpc_config,
        run_hpc_sbatch,
        run_hpc_srun,
        sync_to_hpc,
    )

    config = TestConfig(
        module=module,
        fast=fast,
        hpc_cpus=hpc_cpus,
        hpc_partition=hpc_partition,
        hpc_time=hpc_time,
        hpc_mem=hpc_mem,
    )
    hpc_cfg = _get_hpc_config(config)
    host = hpc_cfg["host"]

    if not _check_ssh(host):
        return {"error": f"Cannot connect to {host}"}

    if not sync_to_hpc(config):
        return {"error": "rsync failed"}

    if async_mode:
        job_id = run_hpc_sbatch(config)
        if job_id:
            return {"job_id": job_id, "host": host}
        return {"error": "Failed to submit job"}
    else:
        exit_code = run_hpc_srun(config)
        return {"exit_code": exit_code, "host": host}


async def test_hpc_poll_handler(
    job_id: str | None = None,
) -> dict[str, Any]:
    """Poll HPC job status.

    Parameters
    ----------
    job_id : str, optional
        Job ID. If None, uses last submitted job.

    Returns
    -------
    dict
        {"state": str, "output": str or None, "job_id": str}
    """
    from scitex._dev._test import poll_hpc_job

    return poll_hpc_job(job_id=job_id)


async def test_hpc_result_handler(
    job_id: str | None = None,
) -> dict[str, Any]:
    """Fetch full HPC test output.

    Parameters
    ----------
    job_id : str, optional
        Job ID. If None, uses last submitted job.

    Returns
    -------
    dict
        {"output": str or None, "job_id": str}
    """
    from scitex._dev._test import fetch_hpc_result

    output = fetch_hpc_result(job_id=job_id)
    return {"output": output, "job_id": job_id or "last"}


async def rename_handler(
    pattern: str,
    replacement: str,
    directory: str = ".",
    confirm: bool = False,
    django_safe: bool = True,
    extra_excludes: list[str] | None = None,
) -> dict[str, Any]:
    """Bulk rename files, contents, directories, and symlinks.

    Two-step safety: call first without confirm to preview,
    then with confirm=True to execute.

    Parameters
    ----------
    pattern : str
        Pattern to search for.
    replacement : str
        String to replace matches with.
    directory : str
        Target directory.
    confirm : bool
        If False (default), dry-run preview. If True, execute.
    django_safe : bool
        Enable Django-safe mode.
    extra_excludes : list of str, optional
        Additional exclude patterns.

    Returns
    -------
    dict
        Rename results with summary.
    """
    from dataclasses import asdict

    from scitex._dev._rename import RenameConfig, bulk_rename
    from scitex._dev._rename._safety import has_uncommitted_changes

    if confirm and has_uncommitted_changes(directory):
        return {"error": "Uncommitted changes detected. Commit or stash first."}

    config = RenameConfig(
        pattern=pattern,
        replacement=replacement,
        directory=directory,
        dry_run=not confirm,
        django_safe=django_safe,
        extra_excludes=extra_excludes or [],
    )
    result = bulk_rename(config)
    return asdict(result)


# EOF
