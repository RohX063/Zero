# ZERO Mistake Punisher v1

## Goal

Improve ZERO's ability to exploit a worsening move by using the search context it already carries instead of introducing a new persistent "punishment mode" or a large collection of ad-hoc flags.

## Reference mechanism

The current official Stockfish search derives an `opponentWorsening` signal from the current and previous static evaluations and uses that context in its hindsight depth adjustment and pruning decisions. ZERO adopts the same *search-context idea* but keeps the implementation small and independent.

## ZERO design

1. `Stack::staticEval` remains the only position-evaluation state required.
2. `Stack::reduction` records the reduction applied to the previous child search so the next node can make a hindsight depth adjustment.
3. `search_mistake.*` owns all mistake-punisher policy.
4. When the opponent-worsening signal is present, ZERO reduces LMR reduction by at most one on the first few candidate moves. This is the only ZERO-specific policy addition in v1.

## Safety

- No new global state.
- No new search mode.
- No changes to move legality.
- No changes to TT semantics.
- No changes to NMP.
- No changes to evaluation.
- The module has a dedicated unit test.

## Validation order

Build → CTest → fixed FEN comparison → tactical regression → engine A/B → only then keep the feature.
