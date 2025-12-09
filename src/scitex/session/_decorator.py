#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-05"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/session/_decorator.py
# ----------------------------------------
"""Session decorator for scitex.

Provides @stx.session decorator that automatically:
- Generates CLI from function signature
- Manages session lifecycle
- Handles errors
- Organizes outputs
"""

import functools
import inspect
import argparse
from pathlib import Path
from typing import Callable, Any, get_type_hints
import sys as sys_module

from ._lifecycle import start, close
from scitex.logging import getLogger
from scitex import INJECTED

# Internal logger for the decorator itself
_decorator_logger = getLogger(__name__)


def session(
    func: Callable = None,
    *,
    verbose: bool = False,
    agg: bool = True,
    notify: bool = False,
    sdir_suffix: str = None,
    **session_kwargs,
) -> Callable:
    """Decorator to wrap function in scitex session.

    Automatically handles:
    - CLI argument parsing from function signature
    - Session initialization (logging, output directories)
    - Execution
    - Cleanup
    - Error handling

    This decorator is designed for script entry points. The decorated function
    should be called without arguments from `if __name__ == '__main__':` to
    trigger CLI parsing and session management.

    Args:
        func: Function to wrap (set automatically by decorator)
        verbose: Enable verbose logging
        agg: Use matplotlib Agg backend
        notify: Send notification on completion
        sdir_suffix: Suffix for output directory name
        **session_kwargs: Additional session configuration parameters

    Example:
        @stx.session
        def analyze(data_path: str, threshold: float = 0.5):
            '''Analyze data file.'''
            data = stx.io.load(data_path)
            result = process(data, threshold)
            stx.io.save(result, "output.csv")
            return 0

        if __name__ == '__main__':
            analyze()  # No arguments = CLI mode with session management

        # CLI: python script.py --data-path data.csv --threshold 0.7

    Example with options:
        @stx.session(verbose=True, notify=True)
        def train_model(model_name: str, epochs: int = 10):
            '''Train ML model.'''
            # These are automatically available as globals:
            # - CONFIG: Session configuration dict
            # - plt: Matplotlib pyplot (configured for session)
            # - COLORS: Custom Colors
            # - rng_manager: RandomStateManager (fixes seeds, creates named generators)
            logger.info(f"Session ID: {CONFIG['ID']}")
            logger.info(f"Output directory: {CONFIG['SDIR_RUN']}")
            # ... training code ...
            return 0

        if __name__ == '__main__':
            train_model()

    Notes:
        - Function name can be anything (not just 'main')
        - Calling with arguments bypasses session management: analyze('/path', 0.5)
        - Only one session-managed function per script
        - Do NOT call multiple @session decorated functions from one script
        - Do NOT nest session-decorated function calls without arguments

    Injected Global Variables:
        When called without arguments (CLI mode), these are injected into globals:
        - CONFIG (dict): Session configuration with ID, SDIR, paths, etc.
        - plt (module): matplotlib.pyplot configured with session settings
        - COLORS (CustomColors): Custom Colors for consistent plotting
        - rng_manager (RandomStateManager): Manages reproducibility by fixing global seeds
                                             and creating named generators via rng_manager("name")
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If called with arguments (not CLI), run directly
            if args or kwargs:
                return func(*args, **kwargs)

            # Otherwise, parse CLI and run with session management
            return _run_with_session(
                func,
                verbose=verbose,
                agg=agg,
                notify=notify,
                sdir_suffix=sdir_suffix,
                **session_kwargs,
            )

        # Store original function for direct access
        wrapper._func = func
        wrapper._is_session_wrapped = True

        return wrapper

    # Handle @stx.session vs @stx.session()
    if func is None:
        # Called with arguments: @stx.session(verbose=True)
        return decorator
    else:
        # Called without arguments: @stx.session
        return decorator(func)


def _run_with_session(
    func: Callable,
    verbose: bool,
    agg: bool,
    notify: bool,
    sdir_suffix: str,
    **session_kwargs,
) -> Any:
    """Run function with full session management."""

    # Get calling file
    frame = inspect.currentframe()
    caller_frame = frame.f_back.f_back  # Go up two levels
    caller_file = caller_frame.f_globals.get("__file__", "unknown.py")

    # Generate argparse from function signature
    parser = _create_parser(func)
    args = parser.parse_args()

    # Clean up INJECTED sentinels from args before passing to session
    cleaned_args = argparse.Namespace(
        **{k: v for k, v in vars(args).items() if not isinstance(v, type(INJECTED))}
    )

    # Start session
    import matplotlib.pyplot as plt

    CONFIG, stdout, stderr, plt, COLORS, rng_manager = start(
        sys=sys_module,
        plt=plt,
        args=cleaned_args,
        file=caller_file,
        sdir_suffix=sdir_suffix or func.__name__,
        verbose=verbose,
        agg=agg,
        **session_kwargs,
    )

    # Create a logger for the user's script
    script_logger = getLogger(func.__module__)

    # Store session variables in function globals
    func_globals = func.__globals__
    func_globals["CONFIG"] = CONFIG
    func_globals["plt"] = plt
    func_globals["COLORS"] = COLORS
    func_globals["rng_manager"] = rng_manager
    func_globals["logger"] = script_logger

    # Log injected globals for user awareness
    _decorator_logger.info("=" * 60)
    _decorator_logger.info("Injected Global Variables (available in your function):")
    _decorator_logger.info("  • CONFIG - Session configuration dict")
    _decorator_logger.info(f"      - CONFIG['ID']: {CONFIG['ID']}")
    _decorator_logger.info(f"      - CONFIG['SDIR_RUN']: {CONFIG['SDIR_RUN']}")
    _decorator_logger.info(f"      - CONFIG['PID']: {CONFIG['PID']}")
    _decorator_logger.info("  • plt - matplotlib.pyplot (configured for session)")
    _decorator_logger.info("  • COLORS - CustomColors (for consistent plotting)")
    _decorator_logger.info("  • rng_manager - RandomStateManager (for reproducibility)")
    _decorator_logger.info("  • logger - SciTeX logger (configured for your script)")
    _decorator_logger.info("=" * 60)

    # Run function
    exit_status = 0
    result = None

    try:
        # Convert args namespace to kwargs
        kwargs = vars(args)

        # Get function parameters
        sig = inspect.signature(func)
        func_params = set(sig.parameters.keys())

        # Map of injected variable names to their actual objects
        injection_map = {
            "CONFIG": CONFIG,
            "plt": plt,
            "COLORS": COLORS,
            "rng_manager": rng_manager,
            "logger": script_logger,
        }

        # Build filtered_kwargs with user args and injected values
        filtered_kwargs = {}

        # First, add all parsed CLI arguments
        for k, v in kwargs.items():
            if k in func_params:
                filtered_kwargs[k] = v

        # Then, inject parameters that have INJECTED as default
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                if isinstance(param.default, type(INJECTED)):
                    # This parameter should be injected
                    if param_name in injection_map:
                        filtered_kwargs[param_name] = injection_map[param_name]

        # Log injected arguments summary
        args_summary = {k: type(v).__name__ for k, v in filtered_kwargs.items()}
        _decorator_logger.info(f"Running {func.__name__} with injected parameters:")
        _decorator_logger.info(args_summary, pprint=True, indent=2)

        # Execute function
        result = func(**filtered_kwargs)

        # Handle return value
        if isinstance(result, int):
            exit_status = result
        else:
            exit_status = 0

    except Exception as e:
        _decorator_logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
        exit_status = 1
        raise

    finally:
        # Close session with error handling
        try:
            close(
                CONFIG=CONFIG,
                verbose=verbose,
                notify=notify,
                message=f"{func.__name__} completed",
                exit_status=exit_status,
            )
        except SystemExit:
            # Allow normal exits
            raise
        except KeyboardInterrupt:
            # Allow Ctrl+C
            raise
        except Exception as e:
            # Log but don't crash on cleanup errors
            try:
                _decorator_logger.error(f"Session cleanup error: {e}")
            except:
                print(f"Session cleanup error: {e}")

        # Final matplotlib cleanup (belt and suspenders approach)
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except:
            pass

    return result


def _create_parser(func: Callable) -> argparse.ArgumentParser:
    """Create ArgumentParser from function signature.

    Args:
        func: Function to create parser for

    Returns:
        Configured ArgumentParser
    """

    # Get function info
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or f"Run {func.__name__}"

    # Try to get type hints
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}

    # Get actual values for deterministic items
    # Get calling file from the decorated function's module
    caller_file = func.__globals__.get("__file__", "unknown.py")

    # Calculate SDIR_OUT (base output directory)
    import os

    sdir_out = Path(os.path.splitext(caller_file)[0] + "_out")
    sdir_run_example = sdir_out / "RUNNING" / "<SESSION_ID>"

    # Get current PID
    current_pid = os.getpid()

    # Check for config YAML files and list all variables with values
    config_status = ""
    try:
        config_dir = Path("./config")
        if config_dir.exists():
            yaml_files = sorted(config_dir.glob("*.yaml"))
            if yaml_files:
                config_status = "        CONFIG from YAML files:\n"

                # Load and list all config variables with their values
                try:
                    import yaml

                    all_vars = []
                    for yaml_file in yaml_files:
                        with open(yaml_file, "r") as f:
                            data = yaml.safe_load(f)
                            if isinstance(data, dict):
                                namespace = yaml_file.stem.upper()
                                for key, value in data.items():
                                    # Format value for display (truncate if too long)
                                    value_str = str(value)
                                    if len(value_str) > 50:
                                        value_str = value_str[:47] + "..."
                                    all_vars.append(
                                        f"        - CONFIG.{namespace}.{key} (from ./config/{yaml_file.name})\n            {value_str}"
                                    )

                    if all_vars:
                        config_status += "\n".join(all_vars)
                    else:
                        config_status = "        CONFIG from YAML files:\n        (no variables found)"
                except Exception as e:
                    # If we can't load the YAML files, just show error
                    config_status = "        CONFIG from YAML files:\n        (unable to load at help-time, will be available at runtime)"
            else:
                config_status = (
                    "        CONFIG from YAML files:\n        (no .yaml files found)"
                )
        else:
            config_status = "        CONFIG from YAML files:\n        (./config/ directory not found)"
    except:
        config_status = (
            "        CONFIG from YAML files:\n        (unable to check at help-time)"
        )

    # Get available color keys
    try:
        from scitex.plt.utils._configure_mpl import configure_mpl
        import matplotlib.pyplot as plt_temp

        _, colors_dict = configure_mpl(plt_temp)
        # Show all color keys
        sorted_keys = sorted(colors_dict.keys())
        color_keys = ", ".join(f"'{k}'" for k in sorted_keys)
    except Exception as e:
        # Fallback if configure_mpl fails
        color_keys = "'blue', 'red', 'green', 'yellow', 'purple', 'orange', ..."

    # Create parser with epilog documenting injected globals with actual values
    epilog = f"""
Global Variables Injected by @session Decorator:

    CONFIG (DotDict)
        Session configuration with ID, paths, timestamps
        Access: CONFIG['key'] or CONFIG.key (both work!)

        - CONFIG.ID
            <SESSION_ID> (created at runtime, e.g., '2025Y-11M-18D-07h53m37s_Z5MR')
        - CONFIG.FILE
            {Path(caller_file)}
        - CONFIG.SDIR_OUT
            {sdir_out}
        - CONFIG.SDIR_RUN
            {sdir_run_example}
        - CONFIG.PID
            {current_pid} (current Python process)
        - CONFIG.ARGS
            {{'arg1': '<value>'}} (parsed from command line)

{config_status}

    plt (module)
        matplotlib.pyplot configured for session

    COLORS (DotDict)
        Color palette for consistent plotting
        Access: COLORS.blue or COLORS['blue'] (both work!)

        Available keys:
            {color_keys}

        Usage:
            plt.plot(x, y, color=COLORS.blue)
            plt.plot(x, y, color=COLORS['blue'])

    rng_manager (RandomStateManager)
        Manages reproducible randomness

    logger (SciTeXLogger)
        Logger configured for your script
"""

    parser = argparse.ArgumentParser(
        description=doc,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add arguments from function signature (skip injected parameters)
    # Track used short forms to avoid conflicts
    used_short_forms = {"h"}  # Reserve -h for help

    for param_name, param in sig.parameters.items():
        # Skip parameters with INJECTED as default (these are injected by decorator)
        if param.default != inspect.Parameter.empty:
            if isinstance(param.default, type(INJECTED)):
                continue  # Skip injected parameters

        # Generate short form
        short_form = _generate_short_form(param_name, used_short_forms)
        if short_form:
            used_short_forms.add(short_form)

        _add_argument(parser, param_name, param, type_hints, short_form)

    return parser


def _generate_short_form(param_name: str, used_short_forms: set) -> str:
    """Generate a short form for a parameter name avoiding conflicts.

    Args:
        param_name: Full parameter name
        used_short_forms: Set of already used short forms

    Returns:
        Short form character or None if no unique form can be generated
    """
    # Strategy 1: Try first letter
    first_letter = param_name[0].lower()
    if first_letter not in used_short_forms:
        return first_letter

    # Strategy 2: Try first letter of each word (for snake_case or camelCase)
    words = param_name.replace("_", " ").replace("-", " ").split()
    if len(words) > 1:
        acronym = "".join(w[0].lower() for w in words)
        if len(acronym) == 1 and acronym not in used_short_forms:
            return acronym

    # Strategy 3: Try first two letters
    if len(param_name) >= 2:
        two_letters = param_name[:2].lower()
        if two_letters not in used_short_forms:
            return two_letters

    # Strategy 4: Try each character in sequence
    for char in param_name.lower():
        if char.isalnum() and char not in used_short_forms:
            return char

    # Give up if no unique short form found
    return None


def _add_argument(
    parser: argparse.ArgumentParser,
    param_name: str,
    param: inspect.Parameter,
    type_hints: dict,
    short_form: str = None,
):
    """Add single argument to parser.

    Args:
        parser: ArgumentParser to add to
        param_name: Parameter name
        param: Parameter object
        type_hints: Type hints dictionary
        short_form: Optional short form (e.g., 'a' for -a)
    """
    from typing import get_origin, get_args, Literal

    # Get type
    param_type = type_hints.get(param_name, param.annotation)
    if param_type == inspect.Parameter.empty:
        param_type = str

    # Get default
    has_default = param.default != inspect.Parameter.empty
    default = param.default if has_default else None

    # Convert parameter name to CLI format
    arg_name = f"--{param_name.replace('_', '-')}"

    # Build argument names list (long form, optionally short form)
    arg_names = [arg_name]
    if short_form:
        arg_names.insert(0, f"-{short_form}")

    # Check for Literal type (choices)
    choices = None
    origin = get_origin(param_type)
    if origin is Literal:
        choices = list(get_args(param_type))
        param_type = type(choices[0]) if choices else str

    # Handle different types
    if param_type == bool:
        # Boolean flags
        parser.add_argument(
            *arg_names,
            action="store_true" if not default else "store_false",
            default=default,
            help=f"(default: {default})",
        )
    else:
        # Regular arguments
        choices_str = f", choices: {choices}" if choices else ""
        kwargs = {
            "type": param_type,
            "help": f"(default: {default}{choices_str})"
            if has_default
            else f"(required{choices_str})",
        }

        if choices:
            kwargs["choices"] = choices

        if has_default:
            kwargs["default"] = default
        else:
            kwargs["required"] = True

        parser.add_argument(*arg_names, **kwargs)


def run(func: Callable, parse_args: Callable = None, **session_kwargs) -> Any:
    """Run function with session management.

    Alternative to decorator for more explicit control.

    Args:
        func: Function to run
        parse_args: Optional custom argument parser
        **session_kwargs: Session configuration

    Example:
        def main(args):
            # Your code
            return 0

        if __name__ == '__main__':
            stx.session.run(main)
    """

    if parse_args is None:
        # Auto-generate parser
        parser = _create_parser(func)
        args = parser.parse_args()
    else:
        # Use custom parser
        args = parse_args()

    # Get file
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    caller_file = caller_frame.f_globals.get("__file__", "unknown.py")

    # Start session
    import matplotlib.pyplot as plt

    CONFIG, stdout, stderr, plt, COLORS, rng_manager = start(
        sys=sys_module,
        plt=plt,
        args=args,
        file=caller_file,
        **session_kwargs,
    )

    # Run
    try:
        if hasattr(args, "__dict__"):
            exit_status = func(args)
        else:
            exit_status = func()

        exit_status = exit_status or 0

    except Exception as e:
        _decorator_logger.error(f"Error: {e}", exc_info=True)
        exit_status = 1
        raise

    finally:
        close(
            CONFIG=CONFIG,
            exit_status=exit_status,
            **session_kwargs,
        )

    return exit_status


# EOF
