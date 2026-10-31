# ZERO Phase 1 Validation Report

## Build

- Release CMake build: PASS
- `-Wall -Wextra -Wpedantic`: PASS with no compiler warnings in the active target
- Unit/regression tests: 2/2 PASS

## Canonical perft

- Start position d5: 4,865,609
- Kiwipete d4: 4,085,603
- Position 3 d4: 43,238
- Position 4 d3: 9,467
- Position 5 d3: 62,379
- Position 6 d3: 89,890

All matched the canonical reference counts used by this project.

## Search smoke

Depth 1..5 from the starting position completed successfully.

Recorded baseline:

- d1: 20 nodes, best `e2e4`
- d2: 420 nodes, best `e2e4`
- d3: 2,987 nodes, best `e2e4`
- d4: 27,709 nodes, best `g1f3`
- d5: 244,642 nodes, best `e2e4`

This milestone intentionally does not claim a strength increase. It establishes an architecture baseline that can be tested after each future search mechanism is introduced.
