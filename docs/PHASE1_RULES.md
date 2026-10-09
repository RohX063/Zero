# Phase 1 Rules

1. No global `ENABLE_*` switch wall.
2. No root-only heuristic patch unless a measured experiment proves a regression source.
3. No commented-out search mechanisms in the active search tree.
4. No strength mechanism is accepted without an isolated A/B test.
5. `ZERO_REFERENCE` (the original MVP) remains frozen.
6. Every structural change must pass build, perft, and search smoke validation before strength testing.
