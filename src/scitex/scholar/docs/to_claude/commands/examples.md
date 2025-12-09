<!-- ---
!-- Timestamp: 2025-06-06 10:06:00
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/.claude/commands/examples.md
!-- --- -->

$ARGUMENTS

01. Understand the examples guidelines
02. Implement examples under `./examples` in organized subdirectories
03. Never place example files directly in `./examples` root
04. Create descriptive subdirectories: `./examples/descriptive-category/example_xxx.py`
05. Directory structure MUST mirror `./src` (primary), `./scripts`, and `./tests` as appropriate
   e.g, `./src/package-name/path/to/xxx.py`
        `./examples/package-name/path/to/example_xxx.py`
        `./tests/package-name/path/to/example_xxx.py`
   e.g, `./scripts/path/to/xxx.py`
        `./examples/path/to/example_xxx.py`
        `./tests/package-name/path/to/example_xxx.py`6. 
06. Keep examples updated and clean
   Remove obsolete example files, contents
   Rename to reflect the current version
07. Examples MUST include at least one figure for demonstration
08. Figures saved via `scitex.io.save` in gif format with numbering: 
   `./examples/category/example_xxx_out/01_figure_name.gif`
09.  GIF files MUST not be movies but a static image unless user requests
10. Run the implemented examples to save the figures using the scitex framework
    Figures must be clean. Always check figures "visually"
    - [ ] Are figures beautiful?
    - [ ] Are figures show what the example want to claim?
    - [ ] If there is a problem with visual check, identify the root cause and re-implement the examples
    Figures must include meaningful x/y labels and appropriate units.
    e.g., NG: `Frequency Index`, OK: `Frequency [Hz]`
11. Examples should include most simple examples for users to run out of the box
    e.g., `import package-name` instead of direct import
    If importing fails, fix importing problems first
    Do not use try and except int importing
12. Include `README.md` summarizing each example directory

<!-- EOF -->
