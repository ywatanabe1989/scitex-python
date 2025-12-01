#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 21:00:11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/vis/demo_vis_editor.py


"""
Demo: stx.vis.edit() - Interactive Figure Editor

This example demonstrates the visual editor for modifying figure styles
and annotations interactively.

Features:
- Multiple backends (web, dearpygui, qt, tkinter, mpl)
- Auto-detection with graceful degradation
- .manual.json for non-destructive edits
- Works with any saved figure (JSON, PNG, CSV)
"""

import numpy as np
import scitex as stx


def create_sample_figure():
    """Create a sample figure with stx.plt and save it."""
    print("Creating sample figure...")

    # Create figure
    fig, ax = stx.plt.subplots()

    # Plot some data
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x), label="sin(x)", id="sine")
    ax.plot(x, np.cos(x), label="cos(x)", id="cosine")

    ax.set_xyt(x="Time [s]", y="Amplitude [a.u.]", t="Trigonometric Functions")
    ax.legend(frameon=False)

    # Save to output directory
    output_path = "/tmp/demo_editor_figure.png"
    stx.io.save(fig, output_path)
    print(f"  Saved to: {output_path}")
    print(f"  JSON: /tmp/demo_editor_figure.json")
    print(f"  CSV:  /tmp/demo_editor_figure.csv")

    fig.close()
    return output_path


def demo_edit_basic():
    """Basic usage of stx.vis.edit()."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Editor Usage")
    print("=" * 60)

    # Create sample figure first
    output_path = create_sample_figure()

    print("\nLaunching editor...")
    print("  Path: /tmp/demo_editor_figure.json")
    print("  Backend: auto (will select best available)")
    print()

    # Launch editor (auto-selects best backend)
    # stx.vis.edit("/tmp/demo_editor_figure.json")

    print("  [Editor would launch here - uncomment line above to run]")
    print()


def demo_edit_backends():
    """Demonstrate different backends."""
    print("\n" + "=" * 60)
    print("Demo 2: Available Backends")
    print("=" * 60)

    print("\nBackend options (in order of preference):")
    print()
    print("  1. 'web'       - Browser-based editor (Flask)")
    print("                   Modern UI, requires: pip install flask")
    print()
    print("  2. 'dearpygui' - GPU-accelerated desktop")
    print("                   Fast, requires: pip install dearpygui")
    print()
    print("  3. 'qt'        - Rich desktop editor")
    print("                   Requires: pip install PyQt6")
    print()
    print("  4. 'tkinter'   - Built-in Python GUI")
    print("                   Works everywhere (Python standard library)")
    print()
    print("  5. 'mpl'       - Minimal matplotlib editor")
    print("                   Always works (matplotlib only)")
    print()

    # Check what's available
    print("Checking your system...")
    from scitex.vis.editor._edit import _detect_best_backend
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        best = _detect_best_backend()
        if w:
            for warning in w:
                print(f"  Warning: {warning.message}")

    print(f"  Best available backend: '{best}'")
    print()


def demo_edit_with_backend():
    """Force specific backend."""
    print("\n" + "=" * 60)
    print("Demo 3: Using Specific Backend")
    print("=" * 60)

    create_sample_figure()

    print("\nExamples:")
    print()
    print("  # Auto-select best backend")
    print('  stx.vis.edit("/tmp/demo_editor_figure.json")')
    print()
    print("  # Force web editor (opens in browser)")
    print('  stx.vis.edit("/tmp/demo_editor_figure.json", backend="web")')
    print()
    print("  # Force tkinter (always works)")
    print('  stx.vis.edit("/tmp/demo_editor_figure.png", backend="tkinter")')
    print()
    print("  # Start from CSV data only")
    print('  stx.vis.edit("/tmp/demo_editor_figure.csv", backend="mpl")')
    print()


def demo_manual_json():
    """Demonstrate .manual.json workflow."""
    print("\n" + "=" * 60)
    print("Demo 4: Manual Override Workflow")
    print("=" * 60)

    print("\nWorkflow:")
    print()
    print("  1. Create figure with stx.plt")
    print("     → Saves: figure.png, figure.json, figure.csv")
    print()
    print("  2. Edit with stx.vis.edit('figure.json')")
    print("     → Opens interactive editor")
    print("     → Change title, colors, add annotations")
    print("     → Click 'Save'")
    print("     → Creates: figure.manual.json")
    print()
    print("  3. Load with manual overrides:")
    print("     fig, ax = stx.plt.load('figure.json', apply_manual=True)")
    print()
    print("  4. Re-run original script")
    print("     → figure.json updates (data changes)")
    print("     → figure.manual.json preserves your style edits")
    print("     → Manual file includes hash to detect staleness")
    print()

    # Show manual.json structure
    print("Example figure.manual.json:")
    print()
    example = {
        "base_file": "figure.json",
        "base_hash": "a1b2c3d4...",
        "overrides": {
            "title": "My Custom Title",
            "xlabel": "Custom X Label",
            "grid": True,
            "annotations": [
                {"type": "text", "text": "Important!", "x": 0.5, "y": 0.9}
            ],
        },
    }
    import json

    print("  " + json.dumps(example, indent=2).replace("\n", "\n  "))
    print()


def demo_edit_from_different_paths():
    """Show how edit() works with different input paths."""
    print("\n" + "=" * 60)
    print("Demo 5: Input Path Flexibility")
    print("=" * 60)

    create_sample_figure()

    print("\nAll these are equivalent:")
    print()
    print("  # From JSON (recommended)")
    print('  stx.vis.edit("/tmp/demo_editor_figure.json")')
    print()
    print("  # From PNG (auto-finds JSON/CSV)")
    print('  stx.vis.edit("/tmp/demo_editor_figure.png")')
    print()
    print("  # From CSV (for data-only editing)")
    print('  stx.vis.edit("/tmp/demo_editor_figure.csv")')
    print()
    print("  # From manual.json (loads existing overrides)")
    print('  stx.vis.edit("/tmp/demo_editor_figure.manual.json")')
    print()

    print("Works with organized directory structure too:")
    print()
    print("  project/")
    print("  ├── json/figure.json")
    print("  ├── csv/figure.csv")
    print("  └── png/figure.png")
    print()
    print(
        '  stx.vis.edit("project/png/figure.png")  # Finds json/csv automatically'
    )
    print()


def run_interactive_demo():
    """Actually run the editor interactively."""
    print("\n" + "=" * 60)
    print("Interactive Demo: Run Editor")
    print("=" * 60)

    create_sample_figure()

    print("\nReady to launch editor.")
    response = input("Launch editor? [y/N]: ").strip().lower()

    if response == "y":
        print("\nLaunching web editor (close browser tab when done)...")
        stx.vis.edit("/tmp/demo_editor_figure.json", backend="web")
        print("\nEditor closed.")
    else:
        print("\nSkipped. Run manually with:")
        print('  stx.vis.edit("/tmp/demo_editor_figure.json")')


@stx.session
def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print(
        "║"
        + "  stx.vis.edit() - Interactive Figure Editor Demo".center(58)
        + "║"
    )
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    demo_edit_basic()
    demo_edit_backends()
    demo_edit_with_backend()
    demo_manual_json()
    demo_edit_from_different_paths()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("stx.vis.edit(path, backend='auto')")
    print()
    print("  path: figure.json | figure.png | figure.csv | figure.manual.json")
    print()
    print("  backend:")
    print("    'auto'      - Select best available (recommended)")
    print("    'web'       - Browser editor (Flask)")
    print("    'dearpygui' - GPU-accelerated desktop")
    print("    'qt'        - Rich desktop (PyQt/PySide)")
    print("    'tkinter'   - Built-in Python GUI")
    print("    'mpl'       - Minimal matplotlib")
    print()
    print("Saves changes to: figure.manual.json")
    print("Load with: stx.plt.load('figure.json', apply_manual=True)")
    print()

    # Ask to run interactive demo
    print("-" * 60)
    run_interactive_demo()


if __name__ == "__main__":
    main()

# EOF
