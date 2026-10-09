# Removed from Active Phase 1 Search Tree

The following older MVP modules were present in the source archive but were not active in the effective MVP search path:

- Transposition table (`tt.*`)
- Killer moves (`killer.*`)
- History (`history.*`)
- Standalone Zobrist initialization/hash path (`zobrist.*`)
- Standalone quiescence search (`quiescence.*`)
- Empty translation units (`engine.cpp`, `move.cpp`, `evaluate.cpp`)
- Unused opening-book / Polyglot files from the original prototype are also excluded from the Phase 1 active target.

They are intentionally not copied into `src/`. The original archive remains the immutable `ZERO_REFERENCE` control implementation.
