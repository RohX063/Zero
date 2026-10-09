# ZERO V1.7.8 Architecture

```text
Position
├── piece[]                  // 64-square lookup mirror
├── byPiece[]                // per-piece bitboards
├── byColor[]                // white/black occupancy
├── byType[]                 // pawn/knight/bishop/rook/queen/king occupancy
└── StateInfo* state

Move
└── from/to squares + compact flags

Move generation
└── bitboard occupancy + precomputed leaper attacks + sliding rays

Search
└── unchanged Phase-1 recursive search
```

The 64-square lookup array is intentional. Modern bitboard engines commonly keep both square->piece lookup and bitboards because each supports a different hot-path operation. Stockfish's `Position` follows this same general shape: it stores a piece array alongside per-type/per-color bitboards. The bitboards are what now drive ZERO's move generation; the lookup exists for constant-time `piece_on(square)` queries.

This phase does not yet implement magic/PEXT slider tables, fixed `MoveList` buffers, direct legal/evasion generation, incremental Zobrist keys, or a fixed-size TT. Those are separate measured upgrades.
