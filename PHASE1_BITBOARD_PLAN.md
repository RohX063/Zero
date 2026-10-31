# Phase 1 — Bitboard Migration Plan

1. Migrate representation first.
2. Preserve `Position` / `StateInfo` boundary.
3. Keep legal filtering by make/undo as a correctness bridge.
4. Validate complete perft suite.
5. Only after correctness passes, optimize sliding attacks (magic/PEXT-style tables), fixed MoveList, then direct legal/evasion generation.
6. Do not add new search parameters during this migration.
