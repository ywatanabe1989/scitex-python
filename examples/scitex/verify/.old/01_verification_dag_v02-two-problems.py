#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/examples/scitex/verify/01_verification_dag.py

"""
Verification DAG - Comprehensive demonstration of scitex.verify module.

Demonstrates:
- Multiple independent data sources (3 branches)
- Parallel processing branches
- Merge points where branches combine
- All verification states: cache (✓), rerun (✓✓), failed (✗)
- Failure propagation through dependency chain
- Multi-format DAG export (HTML, SVG, PNG)

Pipeline structure:
    source_A.csv ──┐
                   ├──> preprocess_A ──> clean_A.csv ──┐
    config_A.json ─┘                                    │
                                                        ├──> merge_all ──> final.csv ──> analyze ──> report.json
    source_B.csv ──> preprocess_B ──> clean_B.csv ──────┤
                                                        │
    source_C.csv ──> preprocess_C ──> clean_C.csv* ─────┘
                                       (modified -> fail)

Usage:
    python 01_verification_dag.py -h
    python 01_verification_dag.py
    python 01_verification_dag.py --demo_failure=False
    python 01_verification_dag.py --demo_rerun=False

Outputs saved to:
    ./01_verification_dag_out/
"""

from pathlib import Path

import numpy as np
import pandas as pd

import scitex as stx

# Directory for outputs (relative to this script)
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "01_verification_dag_out"


# Branch A: Data with config
@stx.session
def source_a(
    n_samples: int = 200,
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Generate source data A with config."""
    logger.info("Generating source A")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "id": range(n_samples),
            "value": np.random.randn(n_samples) * 10 + 100,
            "category": np.random.choice(["X", "Y"], n_samples),
        }
    )
    stx.io.save(data, "source_A.csv")

    config = {"threshold": 0.5, "method": "mean", "source": "A"}
    stx.io.save(config, "config_A.json")

    logger.info(f"Generated {n_samples} samples for source A")
    return 0


@stx.session
def preprocess_a(
    input_file: str = None,
    config_file: str = None,
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Preprocess data A with config."""
    input_file = input_file or str(OUTPUT_DIR / "source_A.csv")
    config_file = config_file or str(OUTPUT_DIR / "config_A.json")
    logger.info("Preprocessing source A")

    data = stx.io.load(input_file)
    config = stx.io.load(config_file)

    # Apply threshold from config
    data_clean = data[data["value"] > data["value"].quantile(config["threshold"])]
    data_clean["source"] = "A"

    stx.io.save(data_clean, "clean_A.csv")
    logger.info(f"Preprocessed {len(data_clean)} rows from source A")
    return 0


# Branch B: Simple processing
@stx.session
def source_b(
    n_samples: int = 300,
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Generate source data B."""
    logger.info("Generating source B")
    np.random.seed(123)

    data = pd.DataFrame(
        {
            "id": range(n_samples),
            "value": np.random.randn(n_samples) * 5 + 50,
            "category": np.random.choice(["X", "Y", "Z"], n_samples),
        }
    )
    stx.io.save(data, "source_B.csv")

    logger.info(f"Generated {n_samples} samples for source B")
    return 0


@stx.session
def preprocess_b(
    input_file: str = None,
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Preprocess data B."""
    input_file = input_file or str(OUTPUT_DIR / "source_B.csv")
    logger.info("Preprocessing source B")

    data = stx.io.load(input_file)
    data_clean = data[data["value"] > 45]
    data_clean["source"] = "B"

    stx.io.save(data_clean, "clean_B.csv")
    logger.info(f"Preprocessed {len(data_clean)} rows from source B")
    return 0


# Branch C: Will be modified to show failure
@stx.session
def source_c(
    n_samples: int = 250,
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Generate source data C."""
    logger.info("Generating source C")
    np.random.seed(456)

    data = pd.DataFrame(
        {
            "id": range(n_samples),
            "value": np.random.randn(n_samples) * 8 + 75,
            "category": np.random.choice(["X", "Z"], n_samples),
        }
    )
    stx.io.save(data, "source_C.csv")

    logger.info(f"Generated {n_samples} samples for source C")
    return 0


@stx.session
def preprocess_c(
    input_file: str = None,
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Preprocess data C."""
    input_file = input_file or str(OUTPUT_DIR / "source_C.csv")
    logger.info("Preprocessing source C")

    data = stx.io.load(input_file)
    data_clean = data[data["value"] > 70]
    data_clean["source"] = "C"

    stx.io.save(data_clean, "clean_C.csv")
    logger.info(f"Preprocessed {len(data_clean)} rows from source C")
    return 0


# Merge point
@stx.session
def merge_all(
    input_a: str = None,
    input_b: str = None,
    input_c: str = None,
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Merge all three branches."""
    input_a = input_a or str(OUTPUT_DIR / "clean_A.csv")
    input_b = input_b or str(OUTPUT_DIR / "clean_B.csv")
    input_c = input_c or str(OUTPUT_DIR / "clean_C.csv")
    logger.info("Merging all branches")

    data_a = stx.io.load(input_a)
    data_b = stx.io.load(input_b)
    data_c = stx.io.load(input_c)

    merged = pd.concat([data_a, data_b, data_c], ignore_index=True)
    merged["combined_score"] = merged["value"] * 1.1

    stx.io.save(merged, "final.csv")
    logger.info(f"Merged {len(merged)} total rows from 3 sources")
    return 0


# Final analysis
@stx.session
def analyze_final(
    input_file: str = None,
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Analyze merged data and generate report."""
    input_file = input_file or str(OUTPUT_DIR / "final.csv")
    logger.info("Analyzing final merged data")

    data = stx.io.load(input_file)

    # Statistics by source
    stats = (
        data.groupby("source")
        .agg({"value": ["mean", "std", "count"], "combined_score": ["mean", "max"]})
        .round(2)
    )

    stx.io.save(stats.reset_index(), "stats_by_source.csv")

    # Final report
    report = {
        "total_records": len(data),
        "sources": list(data["source"].unique()),
        "categories": list(data["category"].unique()),
        "overall_mean": float(data["value"].mean()),
        "overall_std": float(data["value"].std()),
        "max_combined_score": float(data["combined_score"].max()),
    }
    stx.io.save(report, "report.json")

    logger.info("Analysis complete")
    return 0


def _break_chain():
    """Modify clean_C.csv to simulate verification failure (branch C only)."""
    clean_c = OUTPUT_DIR / "clean_C.csv"
    if clean_c.exists():
        data = pd.read_csv(clean_c)
        data["value"] = data["value"] * 1.001  # Small modification
        data.to_csv(clean_c, index=False)
        print(f"Modified {clean_c.name} to simulate failure (branch C only)")


def _run_pipeline():
    """Run the full complex pipeline."""
    print("=" * 70)
    print("Running Verification DAG Pipeline (3 branches merging)")
    print("=" * 70)

    # Branch A (with config)
    print("\n[Branch A] Generating source A...")
    source_a()
    print("[Branch A] Preprocessing A...")
    preprocess_a()

    # Branch B
    print("\n[Branch B] Generating source B...")
    source_b()
    print("[Branch B] Preprocessing B...")
    preprocess_b()

    # Branch C
    print("\n[Branch C] Generating source C...")
    source_c()
    print("[Branch C] Preprocessing C...")
    preprocess_c()

    # Merge and analyze
    print("\n[Merge] Merging all branches...")
    merge_all()
    print("[Analyze] Final analysis...")
    analyze_final()

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)


def _verify_by_rerun():
    """Record rerun verification for sessions to demonstrate ✓✓ badge."""
    from scitex.verify import get_db, verify_run
    from scitex.verify._chain import VerificationLevel, VerificationStatus

    db = get_db()
    runs = db.list_runs(limit=20)

    # Find sessions from branch A (outputs: source_A.csv, clean_A.csv)
    # These won't be affected by break_chain() which only modifies clean_C.csv
    verified_count = 0
    for run in runs:
        sid = run["session_id"]
        outputs = db.get_file_hashes(sid, role="output")
        output_names = [p.split("/")[-1] for p in outputs.keys()]

        # Target branch A sessions
        if any(
            f in output_names for f in ["source_A.csv", "config_A.json", "clean_A.csv"]
        ):
            result = verify_run(sid)
            if result.is_verified:
                db.record_verification(
                    session_id=sid,
                    level=VerificationLevel.RERUN.value,
                    status=VerificationStatus.VERIFIED.value,
                )
                print(f"  ✓✓ Recorded rerun verification: {output_names}")
                verified_count += 1
                if verified_count >= 2:
                    break


def _visualize():
    """Generate visualization in multiple formats."""
    from scitex.verify import render_dag

    target = str(OUTPUT_DIR / "report.json")
    title = "Verification DAG (Multi-branch Pipeline)"

    for fmt in ["html", "svg", "png"]:
        output_path = OUTPUT_DIR / f"dag.{fmt}"
        try:
            render_dag(
                str(output_path), target_file=target, title=title, show_hashes=True
            )
            print(f"Generated: {output_path.name}")
        except Exception as e:
            print(f"Skipped {fmt}: {e}")


@stx.session
def main(
    demo_rerun: bool = True,
    demo_failure: bool = True,
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Verification DAG demo showing cache (✓), rerun (✓✓), and failed (✗) states."""
    # 1. Run the full pipeline
    _run_pipeline()

    # 2. Demo rerun verification (✓✓) FIRST while all files are intact
    if demo_rerun:
        print("\n[Demo] Running rerun verification to show ✓✓ badge...")
        _verify_by_rerun()

    # 3. Demo failure state (✗) by modifying a file
    if demo_failure:
        print("\n[Demo] Modifying branch C file to show ✗ (failure) badge...")
        _break_chain()

    # 4. Generate visualization showing all states
    _visualize()

    print("\n" + "=" * 70)
    print("Verification states demonstrated:")
    print("  ✓   Cache-verified (fast hash check)")
    print("  ✓✓  Rerun-verified (re-executed script)")
    print("  ✗   Failed (hash mismatch after modification)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()


# EOF
