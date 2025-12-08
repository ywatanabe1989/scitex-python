<!-- ---
!-- Timestamp: 2025-06-04 07:43:16
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/.dotfiles/.claude/commands/scitex.md
!-- --- -->

$ARGUMENTS

1. Understand the scitex guidelines. 
   `./docs/to_claude/guidelines/python/*scitex*.py`
2. Check and revise the codebase to STRICTLY FOLLOW THE SciTeX RULES.

## SciTeX Checklist
- In .py file
  - [ ] Added heading docstring
        Example:
        ```python
        """
        Functionalities:
          - Demonstrates Hilbert transform for extracting instantaneous phase and amplitude
          - Shows phase-amplitude coupling (PAC) analysis workflow
          - Visualizes phase-amplitude relationships
          - Demonstrates batch processing and GPU acceleration

        Dependencies:
          - scripts: None
          - packages: numpy, torch, matplotlib, gpac, scitex

        IO:
          - input-files: None (generates synthetic signal with theta-gamma PAC)
          - output-files: hilbert_transform_example.png
        """
        ```
  - [ ] Imported `scitex` and `argparse`
  - [ ] Added `CONFIG = scitex.io.load_configs()`
  - [ ] Created functions with separated concerns
  - [ ] Plotting functions
    - [ ] Used `scitex.plt.subplots(...)` instead of pure matplotlib
    - [ ] Returned `fig`
    - [ ] Returned `fig` object is saved in `main` function using `scitex.io.save(fig, "./relative/save/path.jpg")`
  - [ ] Did not create any directory
    - [ ] NOT USED `os.path.makedirs` <- THIS IS HANDLED BY `scitex.io.save`
  - [ ] Saved with `scitex.io.save(obj, ./relative/save/path.ext)`
    - [ ] relpath starts from `./` or `../`
    - [ ] If script path is `/path/to/script.py`:
      - [ ] `scitex.io.save(opbj, relpath)` will save to `/path/to/script_out/./relative/save/path.ext`
      - [ ] `scitex.io.save` prints "Saved to: {saved_path}" so that NEVER INCLUDE SUCH A MESSAGE - Keep code clean
      - [ ] `/path/to/script_out/` is automatically creatd
      - [ ] `/path/to/script_out/{RUNNING,FINISHED_SUCCESS,FINISHED_FAILED}/YYYYY-MMM-DDD-HHhmmmsss_<4-digit-ID>/logs/{stderr.log,stdout.log}`
  - [ ] Loaded with `scitex.io.load`
  - [ ] main function: 
    - [ ] Requires argparser object and return exit codes
          Example:
          ```python
          def main (args):
              # MAIN LOGIC HERE, USING WELL SEPARATED FUNCTIONS
              return 0
          ```
  - [ ] Implemented `parse_args`
    - [ ] Removed commented out examples (--var, --flag)
          Example:
          ```python
          def parse_args() -> argparse.Namespace:
              """Parse command line arguments."""
              parser = argparse.ArgumentParser(
                  description="Hilbert transform example for gPAC"
              )
              parser.add_argument(
                  "--fs", type=float, default=250.0, help="Sampling frequency (Hz)"
              )
              parser.add_argument(
                  "--duration", type=float, default=5.0, help="Signal duration (seconds)"
              )
              parser.add_argument(
                  "--theta_freq", type=float, default=6.0, help="Theta frequency (Hz)"
              )
              parser.add_argument(
                  "--gamma_freq", type=float, default=40.0, help="Gamma frequency (Hz)"
              )
              parser.add_argument(
                  "--n_taps", type=int, default=101, help="Number of filter taps"
              )
              parser.add_argument(
                  "--use_gpu",
                  action="store_true",
                  help="Use GPU acceleration if available",
              )
              args = parser.parse_args()
              scitex.str.printc(args, c="yellow")
              return args
          ```

  - [ ] Implemented run_main() without editing anything
        ```python
        def run_main() -> None:
            """Initialize scitex framework, run main function, and cleanup."""
            global CONFIG, CC, sys, plt
     
            import sys
     
            import matplotlib.pyplot as plt
            import scitex
     
            args = parse_args()
     
            # Start scitex framework
            CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.gen.start(
                sys,
                plt,
                args=args,
                file=__FILE__,
                verbose=False,
                agg=True,
            )
     
            # Main
            exit_status = main(args)
     
            # Close the scitex framework
            scitex.gen.close(
                CONFIG,
                verbose=False,
                notify=False,
                message="",
                exit_status=exit_status,
            )
     
        ```

<!-- EOF -->
