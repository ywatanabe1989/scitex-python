<!-- ---
!-- Timestamp: 2025-06-07 07:23:09
!-- Author: ywatanabe
!-- File: /home/ywatanabe/.dotfiles/.claude/commands/finalize.md
!-- --- -->


# Project Finalization Checklist

## Code Quality
- [ ] Remove or update obsolete examples
- [ ] Remove or update obsolete tests without unnecessary skips
- [ ] Ensure naming conventions consistent across:
  - Filenames and directories
  - Function names
  - Variable names  
  - Class names
- [ ] Verify file organization consistency

## Testing
- [ ] Run ./examples/run_examples.sh --clear-outputs
- [ ] Run ./tests/run_examples.sh
- [ ] All tests pass without skips
- [ ] KEEP THE OUTPUT DIRECTORIES

## Documentation
- [ ] Documents minimal with no duplicates or overlaps
- [ ] Agent documents only in ./docs/by_agents or README.md
- [ ] Root directory clean
- [ ] Document placement appropriate
- [ ] Can documentation be compressed

## Open Source Readiness
- [ ] Project clean without unnecessary files
- [ ] Ready for public release
- [ ] No sensitive information exposed

<!-- EOF -->