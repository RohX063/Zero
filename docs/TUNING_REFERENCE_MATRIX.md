# ZERO R3.3 Tuning Reference Matrix

This phase intentionally uses a **70% Stockfish 19 / 30% broader-engine research** reference weighting. The ratio describes architectural influence, not copied code. ZERO's implementation remains original.

## 70% — Stockfish 19 / official testing practice

1. **Stockfish 19 `sf_19` search architecture**
   - Used for search-state discipline, TT context, PV/cut-node distinctions, and the general principle that LMR is a contextual selective-search mechanism.
   - Source: https://github.com/official-stockfish/Stockfish/blob/sf_19/src/search.cpp

2. **Stockfish Fishtest SPSA guidance**
   - Used for parameter clipping, perturbation sizing, and the principle that search parameters should be tuned with realistic time controls rather than treating a shallow fixed-depth benchmark as the final strength test.
   - Source: https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html

## 30% — broader 2025–2026 references

1. **ZeroG**
   - Separates offline evaluation tuning from online SPSA search tuning and explicitly tunes LMR bases/margins with self-play.
   - Source: https://github.com/KristianEkman/ZeroG

2. **Zevra 2.7 (2026 report)**
   - Reports SPSA-tuned LMR/NMP/search parameters and a later context-aware LMR pass. This is treated as an engineering lead, not as independently verified Elo evidence.
   - Source: https://chessengines.blogspot.com/2026/09/chess-engine-zevra-27-ja-windows.html

3. **Rusty Rival 1.0.37 (2026 report)**
   - Reports repeated SPSA runs affecting LMR history divisors and continuation thresholds, reinforcing that LMR should be tuned in the context of move history rather than only by move index.
   - Source: https://chessengines.blogspot.com/2026/03/rusty-rival-1.0.37.html

4. **Black Marlin**
   - Uses history-based LMR adjustments, gentler reductions in PV/improving nodes, and contextual move ordering.
   - Source: https://github.com/jnlt3/blackmarlin

5. **FIDE/Google Efficient Chess AI Challenge solution write-up**
   - Provides a real competitive-engine example combining SPSA, SPRT, and distributed engine testing.
   - Source: https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/fix-the-bugs-fix-the-bugs-solution-write-up-3rd

## Design consequences for ZERO

- Keep the R3.2 logarithmic LMR interaction as the baseline.
- First SPSA stage tunes only three independent coefficients.
- Do not put SPSA inside the hot path. Build the reduction table once after UCI parameter changes.
- Pair `theta+` and `theta-` with the same openings and swapped colors.
- Protect known regressions with an anchor gate.
- Keep a true holdout set outside the optimizer.
- Do not accept a candidate from one game, one opponent, or one metric.
