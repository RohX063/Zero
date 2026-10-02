# ZERO V1.7.8 — Bitboard Foundation

Phase 1 representation migration from the 8x8 mailbox board to a Stockfish-style Position representation:

- `Bitboard` occupancy is now the primary move-generation representation.
- `Position` owns per-piece, per-color and per-piece-type bitboards.
- A compact 64-square `board_[]` remains as a square->piece lookup mirror, matching the general Position design used by modern engines.
- `Move` now stores `from`/`to` squares directly instead of row/column pairs.
- Move generation uses bitboard attack masks for pawns, knights, bishops, rooks, queens and kings.
- Legal filtering still uses make/undo; direct legal/evasion generation remains a later optimization.
- No new search heuristic is enabled in this milestone.

## Design rule

This is a representation migration, not a strength patch. Search behavior should be validated only after correctness is established.
