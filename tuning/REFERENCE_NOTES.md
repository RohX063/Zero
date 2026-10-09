# Phase 1 reference mix

The design goal is approximately 70% mature Stockfish-style search engineering
and 30% broader current engine practice.

Stockfish reference themes:
- reductions vary by PV/non-PV context and search state;
- tuned scalers need verification at longer time controls;
- search constants are validated by large STC/LTC tests rather than tiny local
  games.

Broader references used for design ideas:
- Black Marlin documents history-based reductions, reduced PV/improving nodes,
  and staged move ordering.
- Zevra reports SPSA-tuned LMR/NMP/futility/aspiration parameters followed by
  another SPSA round for context-aware LMR and NMP rebalancing.
- Integral publishes SPSA-tuned search releases and separate STC/LTC testing.
- Recent Lambergar releases describe tuned LMR context adjustments, NMP controls,
  continuation history and other search-parameter work.

These references motivate architecture and experiment design only. Their
reported Elo gains are not treated as guaranteed transferable gains for ZERO.
