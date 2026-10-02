# ZERO V1.7.8 — Bitboard Foundation

ZERO now uses a Stockfish-inspired Position representation without changing the Phase-1 search algorithm.

## Core representation

- `Bitboard` is `uint64_t` with A1 = bit 0.
- `Position` stores per-piece, per-color and per-piece-type bitboards.
- A 64-square `piece[]` lookup mirror remains for O(1) `piece_on(square)` queries. This is intentional and matches the general representation used by modern bitboard engines.
- The old `board_[8][8]` mailbox is gone.
- `Move` uses compact 0..63 source/destination squares.

## Move generation

- Pawn, knight and king attacks use precomputed bitboard masks.
- Bishop, rook and queen attacks use occupancy-aware bitboard rays.
- Castling and en-passant state are preserved in `StateInfo`.
- Legal filtering still uses make/undo as a correctness bridge. Direct pinned/check-aware legal generation is a future measured optimization.

## Search

Search, evaluation, MovePicker scoring and Worker/ThreadPool boundaries are otherwise unchanged from Phase 1.

## Validation

- Startpos d5: 4,865,609
- Kiwipete d4: 4,085,603
- Position 3 d4: 43,238
- Position 4 d3: 9,467
- Position 5 d3: 62,379
- Position 6 d3: 89,890
- Randomized position/make-undo integrity: PASS
- Search parity through depth 5: exact nodes/score/best move match against the array Phase-1 baseline

This milestone is a representation migration, not a strength claim.

## Scope
Introduces a dedicated two-slot killer-move framework for quiet-move ordering.

## Changes
- Added `KillerMoves` module with two moves per search ply.
- Killer moves are learned only on interior beta cutoffs from quiet moves.
- MovePicker now accepts TT move, countermove, and two killer moves.
- Ordering priority for quiet signals is TT > countermove > killer 1 > killer 2 > ordinary quiets, while captures retain tactical priority.
- Killer state is cleared at the start of each root search and on `ucinewgame`.
- No evaluation changes, pruning changes, or new hardcoded evaluation parameters.

## Compatibility
Original V2.1 Countersmove behavior is preserved; V2.2 adds killer ordering as an isolated search-state subsystem.
