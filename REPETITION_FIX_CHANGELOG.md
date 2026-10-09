# ZERO 2.6 R3.3 — Repetition Fix Candidate

Date: 2026-10-06

## Goal
Prevent ZERO from voluntarily selecting a move that immediately creates a third occurrence when a non-repeating move can score better.

## Design
The fix is not a blanket repetition penalty.

- `Rules::moveCausesThreefold()` checks whether a legal candidate move would create the third occurrence.
- Root search scores such a move as an exact draw before PVS/LMR selection can distort it.
- Recursive search applies the same exact-draw shortcut after the candidate move is made.
- First-search semantics are preserved: if the first candidate is a repetition draw, the first non-repetition candidate still receives the full PV window.

Intended ordering: `win > draw > loss`.

## Regression coverage
`tests/repetition_search_test.cpp` reconstructs the forensic queen-check repetition position, verifies the repetition count, verifies that Qh5+ is a third-occurrence move, verifies that g2 breaks the repetition, and verifies that depth-4 root search does not choose the repetition move.

## Validation
Build + CTest: 5/5 PASS.
