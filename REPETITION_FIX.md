# ZERO 2.6 R3.3 — Repetition Fix

## Problem
ZERO could enter a third-occurrence repetition even when a winning continuation existed.

## Fix
The search now recognizes a candidate move that immediately creates the third occurrence and assigns it an exact draw score before LMR/PVS/TT selection can distort the result. The rule lives in `src/repetition.cpp` as `Rules::moveCausesThreefold()` and is used both by the recursive search and root search.

This preserves the intended ordering:

`win > draw > loss`

No blanket repetition penalty was added. A repetition move remains a valid choice when all alternatives are no better than a draw.

## Validation
The exact forensic position from the observed game is covered by `tests/repetition_search_test.cpp`.
