# ZERO 2.6 R3.3 — Step 1 Repetition Telemetry

## Purpose
Temporary forensic instrumentation only. This change does **not** alter search scoring, move ordering, repetition rules, or playing strength.

## Telemetry
`src/uci.cpp` now emits to **stderr**:

```text
[REP-TRACE] after-position root_count=<N> threefold=<0|1> halfmove=<N>
[REP-TRACE] before-go root_count=<N> threefold=<0|1> halfmove=<N>
```

The trace uses the existing `Zero::Rules::repetitionCount(position)` implementation.

## Expected forensic behavior
For the reproduction position used in the repetition investigation:

### Full move history supplied
After replaying one previous cycle and reaching the same position a second time, the root should report:

```text
root_count=2
```

### Fresh FEN with no previous moves
The same board supplied as a standalone FEN should report:

```text
root_count=1
```

## Important
Do **not** interpret `root_count=2` as an automatic draw. It means the current position has occurred twice. A third occurrence would be the threefold threshold.

This step only answers the question:

> Is the historical repetition information reaching ZERO?

The search-level question — whether ZERO should proactively avoid a move that creates the third repetition — is a separate step and is intentionally untouched here.

## Live Chessigma test
Run ZERO under Chessigma and capture stderr. Watch the `[REP-TRACE]` lines immediately before `bestmove`.

For the problematic loop, the critical observation is whether ZERO still reports `root_count=1` when we know the position has already occurred twice.
