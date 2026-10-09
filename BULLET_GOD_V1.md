# ZERO 2.6 R3.3 — Bullet God V1

## Goal
Make ZERO spend scarce clock time where the decision is unstable, while stopping early when the root principal variation has stabilized.

## Design
- Conservative optimum and maximum thinking budgets from UCI clock state.
- Smaller default move horizon under very low clocks.
- Root best-move stability tracking across completed iterative-deepening depths.
- Root score volatility tracking.
- Stable PV + stable score can stop after the optimum budget.
- Unstable PV or score keeps the search alive until the hard budget.
- Fixed `movetime` remains exact: no adaptive extension beyond the requested movetime.

## What this does NOT do
- It does not add a separate "bullet mode".
- It does not alter evaluation.
- It does not alter LMR parameters.
- It does not add a repetition penalty.
- It does not claim that shallow search is magically equivalent to deep search.

The intended principle is: **use less clock when the decision is already stable; spend extra clock when the decision is still moving.**

## Reference mapping
This is a ZERO-native simplification of the current Stockfish time-management idea of using best-move stability and evaluation/search instability when deciding whether to continue iterative deepening. It is not a copy of Stockfish code.


## Initial budget examples

With no explicit `movestogo`, ZERO currently uses a 30-move horizon for normal clocks and a 20-move horizon only when the remaining side clock is 3 seconds or less. The hard maximum is capped by a clock reserve.

This keeps the implementation conservative; the next phase should calibrate the budget against actual 1+0 results before changing these constants.
