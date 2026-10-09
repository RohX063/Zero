# ZERO — Parameter Provenance

Purpose: record where every selective-search constant comes from, so ZERO stays
independently authored. All values below are UCI options (see `src/search_params.h`)
and are meant to be moved by ZERO's own SPSA/SPRT runs.

Rule used for this clean-up: techniques (null-move pruning, LMR, aspiration
windows, history heuristics, ...) are general computer-chess ideas. What must be
ZERO's own are the **numbers and the code**. No constants or code were taken from
another engine's source in this revision; defaults are neutral starting guesses,
chosen for ZERO's eval scale (pawn = 100 cp), and are validated only by ZERO's
own testing.

## Constants replaced in R3.4

| Item | Before (matched an external engine) | Now (ZERO-owned, tunable) |
|---|---|---|
| LMR log-interaction | frozen `(2872/128)^2 = 503` | `LMRLogScale` default 470, range 0..1200 (was untunable) |
| LMR base | 982 | `LMRBase` default 820 |
| LMR history coefficient | -16 (effect was ~0.1 ply: effectively dead) | `LMRHistoryCoeff` default -80, range widened to +/-1024 |
| NMP margin | `staticEval >= beta - 13*depth + 365` | `beta + NMPMarginBase - NMPMarginPerDepth*depth`, defaults 110 / 10 |
| NMP reduction | `3 + depth/3 + min(gap/256, 3)` | `NMPReductionBase + depth/NMPDepthDivisor + min(gap/NMPEvalDivisor, NMPEvalCap)`, defaults 3 / 4 / 300 / 3 |
| NMP verification | depth >= 12, window `3*nullDepth/4` | `NMPVerifyDepth` 10, `NMPVerifyPercent` 65 |
| Hindsight thresholds | 3 / 2 / eval-sum 166 | `HindsightRecoverReduction` 3, `HindsightTrimReduction` 2, `HindsightTrimEvalSum` 150 |
| History bonus | `24 + 10*depth`, cap 640 (~0.4% of table range: barely learned) | `HistoryBonusPerDepth` 110, `HistoryBonusMax` 1400, `HistoryMalusPercent` 60 |
| Move overhead | fixed 8..150 ms reserve | `Move Overhead` option, default 20 ms |

## Defaults are placeholders, not conclusions

Every default above is a starting point. The first tuning pass should be
`NMPMargin*`, `LMRBase`, `LMRLogScale`, `HistoryBonus*`, then the rest. Record the
tuned values and the SPRT run that justified them in the table below.

| Date | Parameter | Old | New | Evidence (games / SPRT bounds / result) |
|---|---|---|---|---|
| | | | | |

## Still to be re-derived in ZERO's own words

Items not yet replaced because the feature does not exist in ZERO yet. When each
is written, add a row here with the author and date:

- reverse futility / futility / late-move pruning / SEE pruning
- razoring, ProbCut, internal iterative reduction
- aspiration windows, singular extensions
- correction history, additional continuation-history plies
