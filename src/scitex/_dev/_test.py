#!/usr/bin/env python3
# Timestamp: 2026-02-14
# File: scitex/_dev/_test.py

"""
Core API for running tests locally and on HPC via Slurm.

Functions
---------
run_local : Run pytest locally via subprocess
sync_to_hpc : rsync project to HPC host
run_hpc_srun : Blocking srun on HPC
run_hpc_sbatch : Async sbatch, returns job ID
poll_hpc_job : Check sacct status
fetch_hpc_result : Fetch full output via scp
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scitex.config import PriorityConfig  # noqa: E402

_RSYNC_EXCLUDES = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".eggs",
    "*.egg-info",
    "dist",
    "build",
    "docs/sphinx/_build",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "*_out",
    "GITIGNORED",
    ".pytest-hpc-output",
]

# HPC defaults (match figrecipe production settings)
_HPC_DEFAULTS = {
    "host": "spartan",
    "cpus": 16,
    "partition": "sapphire",
    "time": "00:20:00",
    "mem": "128G",
    "remote_base": "~/proj",
}


@dataclass
class TestConfig:
    """Configuration for test execution."""

    module: str = ""
    parallel: str = "auto"
    fast: bool = False
    coverage: bool = False
    exitfirst: bool = False
    pattern: str = ""
    changed: bool = False
    last_failed: bool = False
    # HPC (None = resolve via SCITEX_DEV_TEST_* env → default)
    hpc_host: str | None = None
    hpc_cpus: int | None = None
    hpc_partition: str | None = None
    hpc_time: str | None = None
    hpc_mem: str | None = None
    remote_base: str | None = None


def _get_project_info() -> tuple:
    """Auto-detect git root and project name.

    Returns
    -------
    tuple
        (git_root_path, project_name)
    """
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_root = os.getcwd()
    project = os.path.basename(git_root)
    return git_root, project


def _get_hpc_config(config: TestConfig) -> dict[str, Any]:
    """Resolve HPC config via PriorityConfig cascade.

    Priority: CLI flag (direct) → SCITEX_DEV_TEST_* env → default.
    """
    pc = PriorityConfig(env_prefix="SCITEX_DEV_TEST_")
    return {
        "host": pc.resolve("host", config.hpc_host, default=_HPC_DEFAULTS["host"]),
        "cpus": pc.resolve(
            "cpus", config.hpc_cpus, default=_HPC_DEFAULTS["cpus"], type=int
        ),
        "partition": pc.resolve(
            "partition", config.hpc_partition, default=_HPC_DEFAULTS["partition"]
        ),
        "time": pc.resolve("time", config.hpc_time, default=_HPC_DEFAULTS["time"]),
        "mem": pc.resolve("mem", config.hpc_mem, default=_HPC_DEFAULTS["mem"]),
        "remote_base": pc.resolve(
            "remote_base", config.remote_base, default=_HPC_DEFAULTS["remote_base"]
        ),
    }


def _job_id_path(git_root: str) -> Path:
    """Path to persist last HPC job ID."""
    return Path(git_root) / ".last-hpc-job"


def _build_pytest_args(config: TestConfig, git_root: str) -> list[str]:
    """Build pytest command-line arguments."""
    args = [sys.executable, "-m", "pytest"]

    # Test path
    if config.module:
        test_dir = os.path.join(git_root, "tests", "scitex", config.module)
        if not os.path.isdir(test_dir):
            test_dir = os.path.join(git_root, "tests", config.module)
        if not os.path.isdir(test_dir):
            test_dir = os.path.join(git_root, "tests")
        args.append(test_dir)
    else:
        args.append(os.path.join(git_root, "tests"))

    # Parallel
    if config.parallel != "0":
        args.extend(["-n", config.parallel])
        args.extend(["--dist", "loadfile"])

    # Options
    if config.fast:
        args.extend(["-m", "not slow"])
    if config.exitfirst:
        args.append("-x")
    if config.pattern:
        args.extend(["-k", config.pattern])
    if config.last_failed:
        args.append("--lf")
    if config.changed:
        args.append("--testmon")
    if config.coverage:
        args.extend(["--cov", "--cov-report=term-missing"])

    args.append("--tb=short")
    return args


def run_local(config: TestConfig) -> int:
    """Run pytest locally via subprocess.

    Parameters
    ----------
    config : TestConfig
        Test configuration.

    Returns
    -------
    int
        Exit code from pytest.
    """
    git_root, _ = _get_project_info()
    args = _build_pytest_args(config, git_root)
    result = subprocess.run(args, cwd=git_root)
    return result.returncode


def _emit_test_event(
    exit_code: int,
    project: str,
    module: str = "",
    source: str = "local",
    log_tail: str = "",
) -> None:
    """Emit test result via scitex.events.

    Delegates to the general-purpose event bus which handles
    state files (~/.scitex/events/) and optional webhook delivery.
    """
    from scitex.events import emit

    emit(
        "test_complete",
        project=project,
        status="success" if exit_code == 0 else "failure",
        payload={
            "exit_code": exit_code,
            "module": module,
            "log_tail": log_tail[-2000:] if log_tail else "",
        },
        source=source,
    )


def _check_ssh(host: str) -> bool:
    """Check SSH connectivity."""
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", host, "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def sync_to_hpc(config: TestConfig) -> bool:
    """Rsync project to HPC host.

    Parameters
    ----------
    config : TestConfig
        Test configuration with HPC settings.

    Returns
    -------
    bool
        True if sync succeeded.
    """
    git_root, project = _get_project_info()
    hpc = _get_hpc_config(config)

    cmd = ["rsync", "-az", "--delete"]
    for exc in _RSYNC_EXCLUDES:
        cmd.extend(["--exclude", exc])
    cmd.append(f"{git_root}/")
    cmd.append(f"{hpc['host']}:{hpc['remote_base']}/{project}/")

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_hpc_srun(config: TestConfig) -> int:
    """Blocking srun on HPC.

    Parameters
    ----------
    config : TestConfig
        Test configuration with HPC settings.

    Returns
    -------
    int
        Exit code from remote pytest.
    """
    _, project = _get_project_info()
    hpc = _get_hpc_config(config)
    cpus = str(hpc["cpus"])

    test_path = f"tests/{config.module}" if config.module else "tests/"
    pytest_opts = f"-n {cpus} --dist loadfile -x --tb=short"
    if config.fast:
        pytest_opts += " -m 'not slow'"
    if config.coverage:
        pytest_opts += " --cov --cov-report=term-missing"

    remote_cmd = (
        f"cd {hpc['remote_base']}/{project} "
        f"&& pip install -e .[dev] -q --no-deps "
        f"&& python -m pytest {test_path} {pytest_opts}"
    )

    ssh_cmd = [
        "ssh",
        hpc["host"],
        f"bash -lc 'srun "
        f"--partition={hpc['partition']} "
        f"--cpus-per-task={cpus} "
        f"--time={hpc['time']} "
        f"--mem={hpc['mem']} "
        f"--job-name=pytest-{project} "
        f'bash -lc "{remote_cmd}"\'',
    ]

    result = subprocess.run(ssh_cmd)
    return result.returncode


def run_hpc_sbatch(config: TestConfig) -> str | None:
    """Async sbatch, returns job ID.

    Parameters
    ----------
    config : TestConfig
        Test configuration with HPC settings.

    Returns
    -------
    str or None
        Job ID string, or None on failure.
    """
    git_root, project = _get_project_info()
    hpc = _get_hpc_config(config)
    cpus = str(hpc["cpus"])
    remote_out = f"{hpc['remote_base']}/{project}/.pytest-hpc-output"

    test_path = f"tests/{config.module}" if config.module else "tests/"
    pytest_opts = f"-n {cpus} --dist loadfile -x --tb=short"
    if config.fast:
        pytest_opts += " -m 'not slow'"
    if config.coverage:
        pytest_opts += " --cov --cov-report=term-missing"

    remote_cmd = (
        f"cd {hpc['remote_base']}/{project} "
        f"&& pip install -e .[dev] -q --no-deps "
        f"&& python -m pytest {test_path} {pytest_opts}"
    )

    ssh_cmd = [
        "ssh",
        hpc["host"],
        f"bash -lc '"
        f"mkdir -p {remote_out} && "
        f"sbatch --parsable "
        f"--partition={hpc['partition']} "
        f"--cpus-per-task={cpus} "
        f"--time={hpc['time']} "
        f"--mem={hpc['mem']} "
        f"--job-name=pytest-{project} "
        f"--output={remote_out}/%j.out "
        f"--error={remote_out}/%j.err "
        f'--wrap="bash -lc \\"{remote_cmd}\\""\'',
    ]

    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    # Extract numeric job ID from output
    match = re.search(r"(\d+)", result.stdout)
    if not match:
        return None

    job_id = match.group(1)
    _job_id_path(git_root).write_text(job_id + "\n")
    return job_id


def poll_hpc_job(
    job_id: str | None = None,
    hpc_host: str | None = None,
) -> dict[str, Any]:
    """Check sacct status.

    Parameters
    ----------
    job_id : str, optional
        Job ID to poll. If None, reads from .last-hpc-job.
    hpc_host : str, optional
        HPC host name. Defaults to HPC_HOST env var or "spartan".

    Returns
    -------
    dict
        {"state": str, "output": str or None, "job_id": str}
    """
    git_root, project = _get_project_info()
    pc = PriorityConfig(env_prefix="SCITEX_DEV_TEST_")
    host = pc.resolve("host", hpc_host, default=_HPC_DEFAULTS["host"])
    remote_base = pc.resolve("remote_base", None, default=_HPC_DEFAULTS["remote_base"])
    remote_out = f"{remote_base}/{project}/.pytest-hpc-output"

    if not job_id:
        jpath = _job_id_path(git_root)
        if jpath.exists():
            job_id = jpath.read_text().strip()
        if not job_id:
            return {"state": "error", "output": None, "job_id": ""}

    # Query sacct
    ssh_cmd = [
        "ssh",
        host,
        f"bash -lc 'sacct -j {job_id} --format=State --noheader -P | head -1'",
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    raw = result.stdout.strip()
    match = re.search(
        r"(COMPLETED|FAILED|RUNNING|PENDING|TIMEOUT|CANCELLED|OUT_OF_ME)", raw
    )
    state = match.group(1) if match else "UNKNOWN"

    output = None
    if state in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"):
        tmp = f"/tmp/pytest-hpc-{job_id}.out"
        subprocess.run(
            ["scp", "-q", f"{host}:{remote_out}/{job_id}.out", tmp],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.exists(tmp):
            with open(tmp) as f:
                output = f.read()

    return {"state": state, "output": output, "job_id": job_id}


def watch_hpc_job(
    job_id: str | None = None,
    hpc_host: str | None = None,
    interval: int = 15,
) -> dict[str, Any]:
    """Poll HPC job until completion.

    Parameters
    ----------
    job_id : str, optional
        Job ID to watch. If None, reads from .last-hpc-job.
    hpc_host : str, optional
        HPC host name.
    interval : int
        Polling interval in seconds.

    Returns
    -------
    dict
        {"state": str, "output": str or None, "job_id": str}
    """
    import time

    while True:
        info = poll_hpc_job(job_id=job_id, hpc_host=hpc_host)
        state = info.get("state", "UNKNOWN")

        if state in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "error"):
            return info
        if state == "UNKNOWN":
            return info

        # Still running/pending, wait and retry
        time.sleep(interval)


def fetch_hpc_result(
    job_id: str | None = None,
    hpc_host: str | None = None,
) -> str | None:
    """Fetch full output via scp.

    Parameters
    ----------
    job_id : str, optional
        Job ID. If None, reads from .last-hpc-job.
    hpc_host : str, optional
        HPC host name.

    Returns
    -------
    str or None
        Full test output, or None if not found.
    """
    git_root, project = _get_project_info()
    pc = PriorityConfig(env_prefix="SCITEX_DEV_TEST_")
    host = pc.resolve("host", hpc_host, default=_HPC_DEFAULTS["host"])
    remote_base = pc.resolve("remote_base", None, default=_HPC_DEFAULTS["remote_base"])
    remote_out = f"{remote_base}/{project}/.pytest-hpc-output"

    if not job_id:
        jpath = _job_id_path(git_root)
        if jpath.exists():
            job_id = jpath.read_text().strip()
        if not job_id:
            return None

    tmp_out = f"/tmp/pytest-hpc-{job_id}.out"
    tmp_err = f"/tmp/pytest-hpc-{job_id}.err"

    subprocess.run(
        ["scp", "-q", f"{host}:{remote_out}/{job_id}.out", tmp_out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["scp", "-q", f"{host}:{remote_out}/{job_id}.err", tmp_err],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if os.path.exists(tmp_out):
        with open(tmp_out) as f:
            return f.read()
    return None


# EOF
