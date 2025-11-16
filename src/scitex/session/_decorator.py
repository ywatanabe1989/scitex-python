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

logger = getLogger(__name__)


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
            # - CC: Custom Colors
            # - rng_manager: RandomStateManager (fixes seeds, creates named generators)
            logger.info(f"Session ID: {CONFIG['ID']}")
            logger.info(f"Output directory: {CONFIG['SDIR']}")
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
        - CC (CustomColors): Custom Colors for consistent plotting
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
    caller_file = caller_frame.f_globals.get('__file__', 'unknown.py')

    # Generate argparse from function signature
    parser = _create_parser(func)
    args = parser.parse_args()

    # Start session
    import matplotlib.pyplot as plt

    CONFIG, stdout, stderr, plt, CC, rng_manager = start(
        sys=sys_module,
        plt=plt,
        args=args,
        file=caller_file,
        sdir_suffix=sdir_suffix or func.__name__,
        verbose=verbose,
        agg=agg,
        **session_kwargs,
    )

    # Store session variables in function globals
    func_globals = func.__globals__
    func_globals['CONFIG'] = CONFIG
    func_globals['plt'] = plt
    func_globals['CC'] = CC
    func_globals['rng_manager'] = rng_manager

    # Log injected globals for user awareness
    logger.info("=" * 60)
    logger.info("Injected Global Variables (available in your function):")
    logger.info("  • CONFIG - Session configuration dict")
    logger.info(f"      - CONFIG['ID']: {CONFIG['ID']}")
    logger.info(f"      - CONFIG['SDIR']: {CONFIG['SDIR']}")
    logger.info(f"      - CONFIG['PID']: {CONFIG['PID']}")
    logger.info("  • plt - matplotlib.pyplot (configured for session)")
    logger.info("  • CC - CustomColors (for consistent plotting)")
    logger.info("  • rng_manager - RandomStateManager (for reproducibility)")
    logger.info("=" * 60)

    # Run function
    exit_status = 0
    result = None

    try:
        # Convert args namespace to kwargs
        kwargs = vars(args)

        # Get function parameters
        sig = inspect.signature(func)
        func_params = set(sig.parameters.keys())

        # Filter kwargs to only include function parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in func_params
        }

        logger.info(f"Running {func.__name__} with args: {filtered_kwargs}")

        # Execute function
        result = func(**filtered_kwargs)

        # Handle return value
        if isinstance(result, int):
            exit_status = result
        else:
            exit_status = 0

    except Exception as e:
        logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
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
                logger.error(f"Session cleanup error: {e}")
            except:
                print(f"Session cleanup error: {e}")

        # Final matplotlib cleanup (belt and suspenders approach)
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
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

    # Create parser with epilog documenting injected globals
    epilog = """
Global Variables Injected by @session Decorator:
  CONFIG (dict)         Session configuration with ID, paths, timestamps
    - CONFIG['ID']      Unique session identifier (timestamp-based)
    - CONFIG['SDIR']    Session output directory (absolute path)
    - CONFIG['PID']     Process ID
    - CONFIG['ARGS']    Parsed command-line arguments

  plt (module)          matplotlib.pyplot configured for session
  CC (CustomColors)     Consistent color palette for plotting
  rng_manager (obj)     RandomStateManager for reproducible randomness

These variables are automatically available in your decorated function.
Use --help to see this message again.
"""

    parser = argparse.ArgumentParser(
        description=doc,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add arguments from function signature
    for param_name, param in sig.parameters.items():
        _add_argument(parser, param_name, param, type_hints)

    return parser


def _add_argument(
    parser: argparse.ArgumentParser,
    param_name: str,
    param: inspect.Parameter,
    type_hints: dict,
):
    """Add single argument to parser.

    Args:
        parser: ArgumentParser to add to
        param_name: Parameter name
        param: Parameter object
        type_hints: Type hints dictionary
    """

    # Get type
    param_type = type_hints.get(param_name, param.annotation)
    if param_type == inspect.Parameter.empty:
        param_type = str

    # Get default
    has_default = param.default != inspect.Parameter.empty
    default = param.default if has_default else None

    # Convert parameter name to CLI format
    arg_name = f"--{param_name.replace('_', '-')}"

    # Handle different types
    if param_type == bool:
        # Boolean flags
        parser.add_argument(
            arg_name,
            action='store_true' if not default else 'store_false',
            default=default,
            help=f"(default: {default})",
        )
    else:
        # Regular arguments
        kwargs = {
            'type': param_type,
            'help': f"(default: {default})" if has_default else "(required)",
        }

        if has_default:
            kwargs['default'] = default
        else:
            kwargs['required'] = True

        parser.add_argument(arg_name, **kwargs)


def run(
    func: Callable,
    parse_args: Callable = None,
    **session_kwargs
) -> Any:
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
    caller_file = caller_frame.f_globals.get('__file__', 'unknown.py')

    # Start session
    import matplotlib.pyplot as plt

    CONFIG, stdout, stderr, plt, CC, rng_manager = start(
        sys=sys_module,
        plt=plt,
        args=args,
        file=caller_file,
        **session_kwargs,
    )

    # Run
    try:
        if hasattr(args, '__dict__'):
            exit_status = func(args)
        else:
            exit_status = func()

        exit_status = exit_status or 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
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
