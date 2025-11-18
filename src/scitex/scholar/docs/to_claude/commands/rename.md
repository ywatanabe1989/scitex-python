<!-- ---
!-- Timestamp: 2025-05-26 00:34:06
!-- Author: ywatanabe
!-- File: /home/ywatanabe/.dotfiles/.claude/commands/rename.md
!-- --- -->


In this repository, a lot of aliases exist. Please refactor those duplications using the `replace_and_rename.sh`.

1. Read and understand `./docs/to_claude/bin/rename.sh`
   For example, `./docs/to_claude/bin/rename.sh -n ecc-auto-start ecc-auto-response-enable` will replace `ecc-auto-start` to `ecc-auto-response-enable` FROM ALL TEXT CONTENTS AND FILE NAMES.
2. Make plan to renaming
3. Write a script to run the script by looping through replacement pairs.
4. Cleanup unnecessary code and files as the above script only replace string.

<!-- EOF -->