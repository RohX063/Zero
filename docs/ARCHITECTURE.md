# ZERO Phase 1 — Clean Search Architecture

Phase 1 is a structural refactor of the original MVP. It deliberately does not add new search strength mechanisms.

## Current boundaries

- `position.*` — board state, castling/en-passant state, make/undo, attack queries, FEN.
- `move.*` — move identity and move flags only.
- `movegen.*` — pseudo/legal move generation.
- `movepick.*` — move ordering and staged emission boundary; current behavior remains MVP-style MVV-LVA ordering.
- `search.*` — recursive search core and node-local `Search::Stack`.
- `search_root.cpp` — root search and iterative deepening.
- `thread.*` — `WorkerThread` and a single-worker `ThreadPool` shell for future parallel search.
- `evaluation.*` — existing handcrafted evaluation.
- `uci.*` — UCI command loop and game-position state storage.

## Future migration path

The names and ownership boundaries are intentionally compatible with a later Stockfish-inspired evolution:

Position → StateInfo → Search::Stack → Worker → WorkerThread/ThreadPool → MovePicker → Search

Phase 1 does not claim feature parity with Stockfish. It establishes the structure first so later TT, richer history, qsearch, incremental hashing, and parallel workers can be introduced without turning `search.cpp` into a monolith.
