<!-- ---
!-- Timestamp: 2025-05-30 00:24:00
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/.dotfiles/.claude/commands/understand-guidelines.md
!-- --- -->


0. Check the current worktree is appropriate:
   Follow the version control guideline:
   `./docs/to_claude/guidelines/**/*version*control*.md`

Once worktree is confirmed as valid, proceed to the following steps:
1. Understand the project structure
   `tree --gitignore ./{src,scripts,tests,examples,project_management}`
2. Always try to use the `filesystem` mcp as it is faster than `Read`
   e.g., `filesystem:read_multiplefiles`
3. Read and understand relevant guidelines, scripts, and so on:
   e.g., `./docs/to_claude/guidelines/**/*.md`
   e.g., `./docs/to_claude/bin/**`
   e.g., ANY FILES YOU THINK IMPORTANT
5. Summarize what you learned from these steps.
6. Memorize the titles of these guidelines and scripts to recall in anytime.

# Final check
7. List all files you read in this request:
``` plaintext
Read and understand: `/path/to/read/file1`
Read and understand: `/path/to/read/file2`
...

```

8. Summarize what you understood using bullet points

<!-- EOF -->