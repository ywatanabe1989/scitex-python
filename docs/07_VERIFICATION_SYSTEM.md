<!-- ---
!-- Timestamp: 2026-01-28 18:41:01
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-python/docs/07_VERIFICATION_SYSTEM.md
!-- --- -->

# How do you prove a paper is reproducible?

Simple: hash everything.

Every scientific computation has inputs, code, and outputs. If you record them and hash the result, you can verify it later by re-running and comparing.

## The Model

Raw → Unit → Chain → Manuscript

- Raw: Untracked data
- Unit: One computation with a hash (input + code + output)
- Chain: Linked units (data → analysis → figure)
- Manuscript: Paper where all chains verify

## How Hashes Work

```
input + code + parameters → computation → output
                               ↓
              hash(input, output) = a7b3c9...
```

Re-run later. Same hash? Verified. Different? Something changed.

## States

| Symbol | State | Meaning |
|--------|-------|---------|
| ○ | raw | not tracked |
| ◐ | recorded | has recipe, not verified yet |
| ● | verified | re-run matches hash |
| ✗ | broken | hash mismatch |

## Example Chain

```
Raw Data (○)
    ↓
Preprocessing (●)
    ↓
Statistics (●)
    ↓
Figure 1 (●)
    ↓
Manuscript (●) ← all verified
```

This is what "reproducible" should mean. Not "I think I can redo this." But "the system proved it."

<!-- EOF -->
