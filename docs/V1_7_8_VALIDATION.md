# ZERO V1.7.8 Validation

## Representation
- Replaced the 8x8 `board_[8][8]` representation with a 64-square piece lookup plus primary bitboards.
- Added per-piece, per-color and per-piece-type occupancy bitboards.
- `Move` now stores 0..63 source/destination squares.
- Move generation uses bitboard occupancy and attack masks.

## Correctness
Canonical perft results are unchanged:
- Start position d5: 4,865,609
- Kiwipete d4: 4,085,603
- Position 3 d4: 43,238
- Position 4 d3: 9,467
- Position 5 d3: 62,379
- Position 6 d3: 89,890

Randomized make/undo integrity test: PASS.

## Search parity
Compared against the Phase-1 array representation on the start position through depth 5:
- depth 1..5 node counts matched exactly: 20 / 420 / 2,987 / 27,709 / 244,642
- best move: e2e4
- score at depth 5: +85 cp

This confirms the representation migration did not change the search result on the regression position.

## Performance note
On the same environment, the complete perft suite ran in approximately 0.48 s with the bitboard representation versus approximately 0.68 s with the array implementation. This is a single-machine microbenchmark, not an Elo result.
