# ZERO 2.6 R3.3 — Bullet God V1

Date: 2026-10-08

## Added
- Search-time module: `src/search_time.h/.cpp`.
- Conservative optimum/maximum time budget from UCI clock state.
- Root best-move stability counter across iterative-deepening iterations.
- Root score-volatility signal to keep searching unstable positions.
- Stable root lines can stop after the optimum budget; unstable lines can use the hard maximum.
- `search_time_test` regression test.

## Preserved
- Repetition fix from the previous candidate.
- Mistake Punisher V1.
- LMR SPSA coefficients.
- Evaluation and move ordering behavior.

## Validation
- Release CMake build: PASS.
- CTest: 6/6 PASS.
- Basic UCI `movetime` smoke test: PASS.

## Status
Experimental candidate only. Do not promote to the official R3.3 branch until 1+0 validation shows that the time-management change improves quality without increasing time losses or tactical instability.
