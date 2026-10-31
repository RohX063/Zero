# ZERO Search Architecture vNext

This package performs the first safe modularization step for the search core.

## What changed

- `Stack` moved into `src/search_stack.h`.
- Search result/config/stat types moved into `src/search_types.h`.
- Shared search helpers moved into `src/search_helpers.h/.cpp`.
- `CMakeLists.txt` includes the new helper translation unit.
- Search behavior is intentionally unchanged by this refactor.

## What did not change

No new pruning heuristic, evaluation term, NMP rule, LMR formula, TT policy, or move-ordering rule was introduced here.

## Next planned boundaries

1. `NodeType` / search context types
2. search preparation helpers
3. reductions and extensions policy module
4. pruning policy module
5. templated PV/non-PV node search core
6. independent tactical-context / opponent-worsening research

Each boundary should compile and pass the full regression suite before the next boundary is introduced.
