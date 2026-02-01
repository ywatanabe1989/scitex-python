#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/_viz/_json.py
"""JSON graph export for verification DAG."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

PathMode = Literal["name", "relative", "absolute"]


def format_path(path: str, mode: PathMode) -> str:
    """Format path according to display mode."""
    if not path or path == "unknown":
        return "unknown"
    p = Path(path)
    if mode == "name":
        return p.name
    elif mode == "relative":
        try:
            return str(p.relative_to(Path.cwd()))
        except ValueError:
            return str(p)
    else:  # absolute
        return str(p.resolve())


def verify_file_hash(path: str, stored_hash: str) -> bool:
    """Check if file's current hash matches stored hash."""
    from .._hash import hash_file

    try:
        current_hash = hash_file(path)
        return current_hash == stored_hash
    except (FileNotFoundError, OSError):
        return False


def file_to_node_id(filename: str) -> str:
    """Convert filename to valid node ID."""
    return "file_" + filename.replace(".", "_").replace("-", "_").replace(" ", "_")


def generate_dag_json(
    session_id: Optional[str] = None,
    target_file: Optional[str] = None,
    path_mode: PathMode = "name",
) -> Dict[str, Any]:
    """
    Generate JSON representation of verification DAG.

    Uses node-link format compatible with D3.js and other visualization libraries.

    Parameters
    ----------
    session_id : str, optional
        Start from this session
    target_file : str, optional
        Start from session that produced this file
    path_mode : str, optional
        Path display mode: "name", "relative", or "absolute"

    Returns
    -------
    dict
        Graph in node-link format with keys:
        - nodes: list of {id, type, name, status, hash, ...}
        - links: list of {source, target, type}
        - metadata: {generated_at, target_file, ...}
    """
    from .._chain import verify_chain, verify_run
    from .._db import get_db

    db = get_db()
    nodes = []
    links = []
    node_ids = set()

    if target_file:
        chain = verify_chain(target_file)
        chain_ids = [run.session_id for run in chain.runs]
    elif session_id:
        chain_ids = db.get_chain(session_id)
    else:
        chain_ids = []

    if not chain_ids:
        return {
            "nodes": [],
            "links": [],
            "metadata": {"generated_at": datetime.now().isoformat(), "empty": True},
        }

    # Collect runs data
    runs_data = []
    for sid in chain_ids:
        run = db.get_run(sid)
        verification = verify_run(sid)
        inputs = db.get_file_hashes(sid, role="input")
        outputs = db.get_file_hashes(sid, role="output")
        runs_data.append(
            {
                "session_id": sid,
                "run": run,
                "verification": verification,
                "inputs": inputs,
                "outputs": outputs,
            }
        )

    # Process in reverse order (oldest first)
    runs_data = list(reversed(runs_data))

    for i, data in enumerate(runs_data):
        sid = data["session_id"]
        run = data["run"]
        verification = data["verification"]
        inputs = data["inputs"]
        outputs = data["outputs"]

        # Script node
        script_id = f"script_{i}"
        script_path = run.get("script_path", "unknown") if run else "unknown"
        nodes.append(
            {
                "id": script_id,
                "type": "script",
                "name": format_path(script_path, path_mode),
                "path": script_path,
                "session_id": sid,
                "status": "verified" if verification.is_verified else "failed",
                "verified_from_scratch": verification.is_verified_from_scratch,
                "hash": run.get("script_hash") if run else None,
            }
        )
        node_ids.add(script_id)

        # Input file nodes
        for fpath, stored_hash in inputs.items():
            file_id = file_to_node_id(Path(fpath).name)
            if file_id not in node_ids:
                file_ok = verify_file_hash(fpath, stored_hash)
                nodes.append(
                    {
                        "id": file_id,
                        "type": "file",
                        "role": "input",
                        "name": format_path(fpath, path_mode),
                        "path": fpath,
                        "status": "verified" if file_ok else "failed",
                        "hash": stored_hash,
                    }
                )
                node_ids.add(file_id)
            links.append({"source": file_id, "target": script_id, "type": "input"})

        # Output file nodes
        for fpath, stored_hash in outputs.items():
            file_id = file_to_node_id(Path(fpath).name)
            if file_id not in node_ids:
                file_ok = verify_file_hash(fpath, stored_hash)
                nodes.append(
                    {
                        "id": file_id,
                        "type": "file",
                        "role": "output",
                        "name": format_path(fpath, path_mode),
                        "path": fpath,
                        "status": "verified" if file_ok else "failed",
                        "hash": stored_hash,
                    }
                )
                node_ids.add(file_id)
            links.append({"source": script_id, "target": file_id, "type": "output"})

    return {
        "nodes": nodes,
        "links": links,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "target_file": target_file,
            "session_id": session_id,
            "num_runs": len(runs_data),
            "num_files": len([n for n in nodes if n["type"] == "file"]),
        },
    }


# EOF
