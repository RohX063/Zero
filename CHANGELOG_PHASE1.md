# ZERO Phase 1 — Clean Search Core

- Replaced `Board` as the active engine-facing name with `Position`.
- Introduced `StateInfo` as the current/previous state chain used by `doMove()` / `undoMove()`.
- Introduced `Search::Stack` for node-local search context.
- Introduced `Search::Worker` to own search execution and counters.
- Introduced `WorkerThread` + single-worker `ThreadPool` shell for future concurrency.
- Split root search from recursive search.
- Split move ordering into a dedicated `MovePicker` boundary.
- Removed disabled TT/history/killer/zobrist/search scaffolding from the active Phase 1 build.
- Kept the original evaluation and move-generation behavior as the baseline implementation.
- Added a reproducible CMake build and canonical perft regression tests.

No new pruning, evaluation bonuses, tactical guards, or other strength heuristics were added in Phase 1.
