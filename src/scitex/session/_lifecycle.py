#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-16 20:25:05 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/session/_lifecycle.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/session/_lifecycle.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

__FILE__ = __file__

"""Session lifecycle management for SciTeX experiments.

This module contains the start() and close() functions that replace
scitex.session.start() and scitex.session.close() with enhanced session management.
"""

import inspect
import os as _os
import re
import shutil
import sys as sys_module
import time
from datetime import datetime
from glob import glob as _glob
from pathlib import Path
from pprint import pprint
from time import sleep
from typing import Any, Dict, Optional, Tuple, Union

from scitex.logging import getLogger

logger = getLogger(__name__)

import matplotlib

# CRITICAL: Set backend before importing pyplot to avoid tkinter issues in headless/WSL environments
# Check if we're in a headless environment (no DISPLAY) or WSL
import os
import sys
import platform

# Detect headless/WSL environments
is_headless = False
try:
    # Check for WSL
    if "microsoft" in platform.uname().release.lower() or "WSL" in os.environ.get(
        "WSL_DISTRO_NAME", ""
    ):
        is_headless = True
    # Check for no X11 display
    elif not os.environ.get("DISPLAY"):
        is_headless = True
except Exception:
    # Fallback: if on Linux without DISPLAY, assume headless
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        is_headless = True

if is_headless:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt_module

from scitex.dict import DotDict

# Lazy imports moved to functions to avoid circular dependency
# from scitex.io._flush import flush
# from scitex.io._save import save as scitex_io_save
# from scitex.io._load import load
# from scitex.io._load_configs import load_configs
from scitex.plt.utils._configure_mpl import configure_mpl
from scitex.repro._gen_ID import gen_ID
from scitex.repro import RandomStateManager
from scitex.str._clean_path import clean_path
from scitex.str._printc import printc as _printc
from scitex.utils._notify import notify as scitex_utils_notify
from ._manager import get_global_session_manager

# For development code flow analysis
try:
    from scitex.dev._analyze_code_flow import analyze_code_flow
except ImportError:

    def analyze_code_flow(file):
        return "Code flow analysis not available"


def _print_header(
    ID: str,
    PID: int,
    file: str,
    args: Any,
    configs: Dict[str, Any],
    verbose: bool = True,
) -> None:
    """Prints formatted header with scitex version, ID, and PID information.

    Parameters
    ----------
    ID : str
        Unique identifier for the current run
    PID : int
        Process ID of the current Python process
    file : str
        File path of the calling script
    args : Any
        Command line arguments or configuration object
    configs : Dict[str, Any]
        Configuration dictionary to display
    verbose : bool, optional
        Whether to print detailed information, by default True
    """

    if args is not None and hasattr(args, "_get_kwargs"):
        args_str = "Arguments:"
        for arg, value in args._get_kwargs():
            args_str += f"\n    {arg}: {value}"
    else:
        args_str = "Arguments: None"

    _printc(
        (
            f"SciTeX v{_get_scitex_version()} | {ID} (PID: {PID})\n\n"
            f"{file}\n\n"
            f"{args_str}"
        ),
        char="=",
    )

    sleep(1)
    if verbose:
        from pprint import pformat
        config_str = pformat(configs.to_dict())
        logger.info(f"\n{'-' * 40}\n\n{config_str}\n\n{'-' * 40}\n")
    sleep(1)


def _initialize_env(IS_DEBUG: bool) -> Tuple[str, int]:
    """Initialize environment with ID and PID.

    Parameters
    ----------
    IS_DEBUG : bool
        Debug mode flag

    Returns
    -------
    tuple
        (ID, PID) - Unique identifier and Process ID
    """
    ID = gen_ID(N=4) if not IS_DEBUG else "DEBUG_" + gen_ID(N=4)
    PID = _os.getpid()
    return ID, PID


def _setup_configs(
    IS_DEBUG: bool,
    ID: str,
    PID: int,
    file: str,
    sdir: str,
    relative_sdir: str,
    verbose: bool,
) -> Dict[str, Any]:
    """Setup configuration dictionary with basic parameters.

    Parameters
    ----------
    IS_DEBUG : bool
        Debug mode flag
    ID : str
        Unique identifier
    PID : int
        Process ID
    file : str
        File path
    sdir : str
        Save directory path
    relative_sdir : str
        Relative save directory path
    verbose : bool
        Verbosity flag

    Returns
    -------
    dict
        Configuration dictionary
    """
    # Calculate SDIR_OUT (base output directory)
    # sdir format: /path/to/script_out/RUNNING/ID/
    sdir_path = Path(sdir) if sdir else None
    if sdir_path:
        # Remove /RUNNING/ID/ to get base output dir
        parts = sdir_path.parts
        if "RUNNING" in parts:
            running_idx = parts.index("RUNNING")
            sdir_out = Path(*parts[:running_idx])
        else:
            sdir_out = sdir_path.parent
    else:
        sdir_out = None

    # Load YAML configs from ./config/*.yaml
    from scitex.io._load_configs import load_configs

    CONFIGS = load_configs(IS_DEBUG).to_dict()

    # Add session-specific config with clean structure (Path objects only)
    CONFIGS.update(
        {
            "ID": ID,
            "PID": PID,
            "START_DATETIME": datetime.now(),
            "FILE": Path(file) if file else None,
            "SDIR_OUT": sdir_out,
            "SDIR_RUN": sdir_path,
        }
    )
    return CONFIGS


def _setup_matplotlib(
    plt: plt_module = None, agg: bool = False, **mpl_kwargs: Any
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Configure matplotlib settings.

    Parameters
    ----------
    plt : module
        Matplotlib.pyplot module (will be replaced with scitex.plt)
    agg : bool
        Whether to use Agg backend
    **mpl_kwargs : dict
        Additional matplotlib configuration parameters

    Returns
    -------
    tuple
        (plt, COLORS) - Configured scitex.plt module and color cycle
    """
    if plt is not None:
        plt.close("all")
        _, COLORS = configure_mpl(plt, **mpl_kwargs)
        COLORS["gray"] = COLORS["grey"]

        # Note: Backend is now set early in module initialization (line 50)
        # to avoid tkinter threading issues in headless/WSL environments.
        # The 'agg' parameter is kept for backwards compatibility but has
        # no effect since backend must be set before pyplot import.
        if agg and not is_headless:
            logger.warning(
                "agg=True specified but backend was already set to Agg "
                "during module initialization for headless environment"
            )

        # Replace matplotlib.pyplot with scitex.plt to get wrapped functions
        import scitex.plt as stx_plt

        return stx_plt, COLORS
    return plt, None


def _simplify_relative_path(sdir: str) -> str:
    """
    Simplify the relative path by removing specific patterns.

    Example
    -------
    sdir = '/home/user/scripts/memory-load/distance_between_gs_stats/RUNNING/2024Y-09M-12D-02h44m40s_GlBZ'
    simplified_path = simplify_relative_path(sdir)
    print(simplified_path)
    # Output: './memory-load/distance_between_gs_stats/'

    Parameters
    ----------
    sdir : str
        The directory path to simplify

    Returns
    -------
    str
        Simplified relative path
    """
    base_path = _os.getcwd()
    relative_sdir = _os.path.relpath(sdir, base_path) if base_path else sdir
    simplified_path = relative_sdir.replace("scripts/", "./scripts/").replace(
        "RUNNING/", ""
    )
    # Remove date-time pattern and random string
    simplified_path = re.sub(
        r"\d{4}Y-\d{2}M-\d{2}D-\d{2}h\d{2}m\d{2}s_\w+/?$", "", simplified_path
    )
    return simplified_path


def _get_debug_mode() -> bool:
    """Get debug mode from configuration."""
    try:
        from scitex.io._load import load

        IS_DEBUG_PATH = "./config/IS_DEBUG.yaml"
        if _os.path.exists(IS_DEBUG_PATH):
            IS_DEBUG = load(IS_DEBUG_PATH).get("IS_DEBUG", False)
            if IS_DEBUG == "true":
                IS_DEBUG = True
        else:
            IS_DEBUG = False

    except Exception as e:
        print(e)
        IS_DEBUG = False
    return IS_DEBUG


def _clear_python_log_dir(log_dir: str) -> None:
    """Clear Python log directory."""
    try:
        if _os.path.exists(log_dir):
            _os.system(f"rm -rf {log_dir}")
    except Exception as e:
        print(f"Failed to clear directory {log_dir}: {e}")


def _get_scitex_version() -> str:
    """Gets scitex version"""
    try:
        import scitex

        return scitex.__version__
    except Exception as e:
        print(e)
        return "(not found)"


def start(
    sys: sys_module = None,
    plt: plt_module = None,
    file: Optional[str] = None,
    sdir: Optional[Union[str, Path]] = None,
    sdir_suffix: Optional[str] = None,
    args: Optional[Any] = None,
    os: Optional[Any] = None,
    random: Optional[Any] = None,
    np: Optional[Any] = None,
    torch: Optional[Any] = None,
    seed: int = 42,
    agg: bool = False,
    fig_size_mm: Tuple[int, int] = (160, 100),
    fig_scale: float = 1.0,
    dpi_display: int = 100,
    dpi_save: int = 300,
    fontsize="small",
    autolayout=True,
    show_execution_flow=False,
    hide_top_right_spines: bool = True,
    alpha: float = 0.9,
    line_width: float = 1.0,
    clear_logs: bool = False,
    verbose: bool = True,
) -> Tuple[DotDict, Any, Any, Any, Optional[Dict[str, Any]], Any]:
    """Initialize experiment session with reproducibility settings.

    This function replaces scitex.session.start() with enhanced session management.

    Parameters
    ----------
    sys : module, optional
        Python sys module for I/O redirection
    plt : module, optional
        Matplotlib pyplot module for plotting configuration
    file : str, optional
        Script file path. If None, automatically detected
    sdir : Union[str, Path], optional
        Save directory path. Can be a string or pathlib.Path object. If None, automatically generated
    sdir_suffix : str, optional
        Suffix to append to save directory
    args : object, optional
        Command line arguments or configuration object
    os, random, np, torch : modules, optional
        Modules for random seed fixing
    seed : int, default=42
        Random seed for reproducibility
    agg : bool, default=False
        Whether to use matplotlib Agg backend
    fig_size_mm : tuple, default=(160, 100)
        Figure size in millimeters
    fig_scale : float, default=1.0
        Scale factor for figure size
    dpi_display, dpi_save : int
        DPI for display and saving
    fontsize : str, default='small'
        Font size setting
    autolayout : bool, default=True
        Enable matplotlib autolayout
    show_execution_flow : bool, default=False
        Show code execution flow analysis
    hide_top_right_spines : bool, default=True
        Whether to hide top and right spines
    alpha : float, default=0.9
        Default alpha value for plots
    line_width : float, default=1.0
        Default line width for plots
    clear_logs : bool, default=False
        Whether to clear existing log directory
    verbose : bool, default=True
        Whether to print detailed information

    Returns
    -------
    tuple
        (CONFIGS, stdout, stderr, plt, COLORS, rng)
        - CONFIGS: Configuration dictionary
        - stdout, stderr: Redirected output streams
        - plt: Configured matplotlib.pyplot module
        - COLORS: Color cycle dictionary
        - rng: Global RandomStateManager instance for reproducible random generation
    """
    IS_DEBUG = _get_debug_mode()
    ID, PID = _initialize_env(IS_DEBUG)

    # Convert Path objects to strings for internal processing
    if sdir is not None and isinstance(sdir, Path):
        sdir = str(sdir)

    ########################################
    # Defines SDIR (DO NOT MODIFY THIS SECTION)
    ########################################
    if sdir is None:
        # Define __file__
        if file:
            caller_file = file
        else:
            caller_file = inspect.stack()[1].filename
            if "ipython" in __file__:
                caller_file = f"/tmp/{_os.getenv('USER')}.py"

        # Convert to absolute path if relative and resolve symlinks
        if not _os.path.isabs(caller_file):
            caller_file = _os.path.realpath(_os.path.abspath(caller_file))
        else:
            # Even if already absolute, resolve symlinks to get the real path
            caller_file = _os.path.realpath(caller_file)

        # Define sdir
        sdir = clean_path(_os.path.splitext(caller_file)[0] + f"_out/RUNNING/{ID}/")

        # Optional
        if sdir_suffix:
            sdir = sdir[:-1] + f"-{sdir_suffix}/"

    if clear_logs:
        _clear_python_log_dir(sdir + caller_file + "/")
    _os.makedirs(sdir, exist_ok=True)
    relative_sdir = _simplify_relative_path(sdir)
    ########################################

    # Setup configs after having all necessary parameters
    CONFIGS = _setup_configs(IS_DEBUG, ID, PID, file, sdir, relative_sdir, verbose)

    # Logging
    if sys is not None:
        from scitex.io._flush import flush

        flush(sys)
        # Lazy import to avoid circular dependency
        from scitex.logging._Tee import tee

        sys.stdout, sys.stderr = tee(sys, sdir=sdir, verbose=verbose)
        CONFIGS["_sys"] = sys  # Private key, won't show in user-facing pprint

        # Redirect logging handlers to use the tee-wrapped streams
        # This ensures that logger output is captured in the log files
        import logging

        # Update all existing StreamHandler instances to use our wrapped streams
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            try:
                logger = logging.getLogger(logger_name)
                for handler in logger.handlers:
                    if isinstance(handler, logging.StreamHandler):
                        # StreamHandler typically uses stderr by default
                        if not hasattr(handler, "stream"):
                            continue
                        # Check if handler is using the original stderr or stdout
                        if handler.stream in (sys.__stderr__, sys.__stdout__):
                            # Replace with our tee-wrapped stream
                            handler.stream = (
                                sys.stderr
                                if handler.stream == sys.__stderr__
                                else sys.stdout
                            )
            except Exception:
                # Silently skip any logger that can't be updated
                pass

        # Also update the root logger handlers
        try:
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    if not hasattr(handler, "stream"):
                        continue
                    # Check if handler is using the original stderr or stdout
                    if handler.stream in (sys.__stderr__, sys.__stdout__):
                        # Replace with our tee-wrapped stream
                        handler.stream = (
                            sys.stderr
                            if handler.stream == sys.__stderr__
                            else sys.stdout
                        )
        except Exception:
            # Silently skip if root logger can't be updated
            pass

    # Initialize RandomStateManager (automatically fixes all seeds)
    rng = RandomStateManager(seed=seed, verbose=verbose)
    if verbose:
        logger.info(f"Initialized RandomStateManager with seed {seed}")

    # Matplotlib configurations
    plt, COLORS = _setup_matplotlib(
        plt,
        agg,
        fig_size_mm=fig_size_mm,
        fig_scale=fig_scale,
        dpi_display=dpi_display,
        dpi_save=dpi_save,
        hide_top_right_spines=hide_top_right_spines,
        alpha=alpha,
        line_width=line_width,
        fontsize=fontsize,
        autolayout=autolayout,
        verbose=verbose,
    )

    # Adds argument-parsed variables
    if args is not None:
        CONFIGS["ARGS"] = vars(args) if hasattr(args, "__dict__") else args

    CONFIGS = DotDict(CONFIGS)

    # Register session
    session_manager = get_global_session_manager()
    session_manager.create_session(ID, CONFIGS)

    _print_header(ID, PID, file, args, CONFIGS, verbose)

    if show_execution_flow:
        structure = analyze_code_flow(file)
        _printc(structure)

    # Return appropriate values based on whether sys was provided
    if sys is not None:
        return CONFIGS, sys.stdout, sys.stderr, plt, COLORS, rng
    else:
        return CONFIGS, None, None, plt, COLORS, rng


def _format_diff_time(diff_time):
    """Format time difference as HH:MM:SS."""
    total_seconds = int(diff_time.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    diff_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return diff_time_str


def _process_timestamp(CONFIG, verbose=True):
    """Process session timestamps."""
    try:
        CONFIG["END_DATETIME"] = datetime.now()
        CONFIG["RUN_DURATION"] = _format_diff_time(
            CONFIG["END_DATETIME"] - CONFIG["START_DATETIME"]
        )
        if verbose:
            logger.info(
                f"\nSTART TIME: {CONFIG['START_DATETIME']}\n"
                f"END TIME: {CONFIG['END_DATETIME']}\n"
                f"RUN DURATION: {CONFIG['RUN_DURATION']}\n"
            )

    except Exception as e:
        print(e)

    return CONFIG


def _save_configs(CONFIG):
    """Save configuration to files."""
    from scitex.io._save import save as scitex_io_save

    # Convert to dict with all keys (including private ones) for saving
    config_dict = (
        CONFIG.to_dict(include_private=True) if hasattr(CONFIG, "to_dict") else CONFIG
    )

    scitex_io_save(
        config_dict, str(CONFIG["SDIR_RUN"] / "CONFIGS/CONFIG.pkl"), verbose=False
    )
    scitex_io_save(
        config_dict, str(CONFIG["SDIR_RUN"] / "CONFIGS/CONFIG.yaml"), verbose=False
    )


def _escape_ansi_from_log_files(log_files):
    """Remove ANSI escape sequences from log files.

    Parameters
    ----------
    log_files : list
        List of log file paths to clean
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    # ANSI code escape
    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            content = file.read()
        cleaned_content = ansi_escape.sub("", content)
        with open(f, "w", encoding="utf-8") as file:
            file.write(cleaned_content)


def _args_to_str(args_dict):
    """Convert args dictionary to formatted string."""
    if args_dict:
        max_key_length = max(len(str(k)) for k in args_dict.keys())
        return "\n".join(
            f"{str(k):<{max_key_length}} : {str(v)}"
            for k, v in sorted(args_dict.items())
        )
    else:
        return ""


def running2finished(CONFIG, exit_status=None, remove_src_dir=True, max_wait=60):
    """Move session from RUNNING to FINISHED directory.

    Parameters
    ----------
    CONFIG : dict
        Session configuration dictionary
    exit_status : int, optional
        Exit status code (0=success, 1=error, None=finished)
    remove_src_dir : bool, default=True
        Whether to remove source directory after copy
    max_wait : int, default=60
        Maximum seconds to wait for copy operation

    Returns
    -------
    dict
        Updated configuration with new SDIR
    """
    if exit_status == 0:
        dest_dir = str(CONFIG["SDIR_RUN"]).replace("RUNNING/", "FINISHED_SUCCESS/")
    elif exit_status == 1:
        dest_dir = str(CONFIG["SDIR_RUN"]).replace("RUNNING/", "FINISHED_ERROR/")
    else:  # exit_status is None:
        dest_dir = str(CONFIG["SDIR_RUN"]).replace("RUNNING/", "FINISHED/")

    src_dir = str(CONFIG["SDIR_RUN"])
    _os.makedirs(dest_dir, exist_ok=True)
    try:
        # Copy files individually
        for item in _os.listdir(src_dir):
            s = _os.path.join(src_dir, item)
            d = _os.path.join(dest_dir, item)
            if _os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

        start_time = time.time()
        while not _os.path.exists(dest_dir) and time.time() - start_time < max_wait:
            time.sleep(0.1)
        if _os.path.exists(dest_dir):
            print()
            logger.success(
                f"Congratulations! The script completed: {dest_dir}",
            )

            if remove_src_dir:
                shutil.rmtree(src_dir)

            # Cleanup RUNNING when empty
            running_base = os.path.dirname(src_dir.rstrip("/"))
            if os.path.basename(running_base) == "RUNNING":
                try:
                    os.rmdir(running_base)
                except OSError:
                    pass

        else:
            print(f"Copy operation timed out after {max_wait} seconds")

        CONFIG["SDIR_RUN"] = Path(dest_dir)
    except Exception as e:
        print(e)

    finally:
        return CONFIG


def close(CONFIG, message=":)", notify=False, verbose=True, exit_status=None):
    """Close experiment session and finalize logging.

    This function replaces scitex.session.close() with enhanced session management.

    Parameters
    ----------
    CONFIG : DotDict
        Configuration dictionary from start()
    message : str, default=':)'
        Completion message
    notify : bool, default=False
        Whether to send notification
    verbose : bool, default=True
        Whether to print verbose output
    exit_status : int, optional
        Exit status code (0=success, 1=error, None=finished)
    """
    sys = None  # Initialize sys outside try block
    try:
        CONFIG.EXIT_STATUS = exit_status
        CONFIG = CONFIG.to_dict()
        CONFIG = _process_timestamp(CONFIG, verbose=verbose)
        sys = CONFIG.pop("_sys", None)  # Pop private sys reference

        # CRITICAL: Close matplotlib BEFORE closing streams to prevent segfault
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import atexit

            # Close all figures
            plt.close("all")

            # CRITICAL: Unregister matplotlib's atexit handlers to prevent segfault
            # Matplotlib registers handlers that try to cleanup on exit,
            # but we're closing streams first, causing segfault
            try:
                # Remove matplotlib-related atexit handlers
                import weakref

                # Matplotlib uses weakref for some cleanup
                if hasattr(matplotlib, "_pylab_helpers"):
                    matplotlib._pylab_helpers.Gcf.destroy_all()

                # Clear any pyplot state
                if hasattr(plt, "get_fignums"):
                    for fignum in plt.get_fignums():
                        plt.close(fignum)

            except Exception:
                pass

            # Force garbage collection to cleanup matplotlib resources
            import gc

            gc.collect()

            if verbose:
                logger.info("Matplotlib cleanup completed")

        except Exception as e:
            if verbose:
                logger.warning(f"Could not close matplotlib: {e}")

        _save_configs(CONFIG)

        # RUNNING to FINISHED
        CONFIG = running2finished(CONFIG, exit_status=exit_status)

        # ANSI code escape
        log_files = _glob(str(CONFIG["SDIR_RUN"]) + "logs/*.log")
        _escape_ansi_from_log_files(log_files)

        if CONFIG.get("ARGS"):
            message += f"\n{_args_to_str(CONFIG.get('ARGS'))}"

        if notify:
            try:
                message = (
                    f"[DEBUG]\n" + str(message)
                    if CONFIG.get("DEBUG", False)
                    else str(message)
                )
                scitex_utils_notify(
                    message=message,
                    ID=CONFIG["ID"],
                    file=CONFIG.get("FILE"),
                    attachment_paths=log_files,
                    verbose=verbose,
                )
            except Exception as e:
                print(e)

        # Close session
        session_manager = get_global_session_manager()
        session_manager.close_session(CONFIG["ID"])

    finally:
        # Only close if they're custom file objects (Tee objects)
        if sys:
            try:
                # First, flush all outputs
                if hasattr(sys, "stdout") and hasattr(sys.stdout, "flush"):
                    sys.stdout.flush()
                if hasattr(sys, "stderr") and hasattr(sys.stderr, "flush"):
                    sys.stderr.flush()

                # Then close Tee objects
                if hasattr(sys, "stdout") and hasattr(sys.stdout, "_log_file"):
                    sys.stdout.close()
                if hasattr(sys, "stderr") and hasattr(sys.stderr, "_log_file"):
                    sys.stderr.close()
            except Exception:
                # Silent fail to ensure logs are saved even if there's an error
                pass


# EOF
