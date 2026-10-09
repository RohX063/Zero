# ZERO R3.4 — Bug fixes and constant clean-up

Base: `ZERO_2.6_R3.3_Bullet_God_V1_Candidate`.
Scope: correctness bugs found in audit + replacing externally-matching constants
with ZERO-owned, tunable parameters. No new pruning features yet (that is the next phase).

## Bugs fixed

| # | Bug | Fix |
|---|---|---|
| 1 | `go depth N`, `go nodes`, `go infinite` stopped after depth 1 (`optimumMs <= 0` returned "stop") | Unbounded budget keeps deepening; a search with no depth and no clock gets a finite default depth of 16 |
| 2 | TT cutoffs were also taken at PV nodes (docs said they weren't) | TT cutoff only when `!pvNode`, and not when the halfmove clock is >= 90 |
| 3 | Non-PV re-search repeated the identical zero-window search at the same depth | Re-search only after a reduced probe fails high (same window); full-window re-search only at PV nodes |
| 4 | Node types never alternated: every zero-window child was a cut node | First child of PV = PV; other children of PV = cut; below non-PV nodes the type alternates; reduced probes are cut |
| 5 | "Mistake punisher" read the reduction used to reach the *parent*, one ply off | Parent publishes its applied reduction in its own stack slot; child reads and clears it |
| 6 | `improving` / opponent-worsening compared against stale `0` evals (in check, shallow nodes) | `Stack::evalValid`; all comparisons require valid evals on both sides |
| 7 | Counter-move updated on every node (including fail-low and captures); fail-low best move got a positive history bonus | Counter-move only on a quiet beta cutoff; history updates only if some move beat alpha |
| 8 | Draws: threefold-only inside the tree, fifty-move rule unscored (75-move auto), no insufficient material, repetition chain walked twice per node | `Rules::isDrawInSearch`: one repetition inside the tree, rule-50 at 100 (not when in check), K vs K / K+minor vs K; checked once per node, also at the horizon |
| 9 | Stop flag not sticky (relied on the node counter staying frozen while unwinding) | `stopped_` latches once the deadline is seen |
| 10 | Root never wrote itself to the TT, so previous-iteration best move was not searched first | `rootBestMove_` anchors root move ordering |
| 11 | Partial iteration discarded (about 25% of a 3 s budget unused in testing) | If the first root move (previous best) finished, its result is used when time runs out |
| 12 | Increment larger than remaining clock produced a target beyond the safe maximum (flag risk) | Target is capped to 80% of (clock - reserve) |
| 13 | Mates printed as `cp 99999`; mated root printed a garbage score; the SPSA harness therefore scored every checkmate as a draw | `score mate N`, `info depth 0 score mate 0` / `cp 0` for terminal roots, `pv` in info lines |
| 14 | `bestmove 0000` possible when time ran out before depth 1 finished | Falls back to first legal move |
| 15 | Default CMake build had no optimization flags | Defaults to Release |
| 16 | Unconditional +1 for every check and promotion | Rationed by `ExtensionPlyLimitPct` and switchable (`CheckExtension`, `PromotionExtension`) |
| 17 | Mate-distance pruning documented but absent | Added |
| 18 | `[REP-TRACE]` forensic output written to stderr on every position/go | Only when env `ZERO_REP_TRACE` is set |
| 19 | `repetition_search_test` failed on the original upload (its expected threefold never occurs in its own move sequence) | Test corrected; tests added for search-tree repetition, insufficient material, rule-50, time-budget regressions |

## New UCI options

`Move Overhead`, `LMRLogScale`, `NMPMinDepth`, `NMPMarginBase`, `NMPMarginPerDepth`,
`NMPReductionBase`, `NMPDepthDivisor`, `NMPEvalDivisor`, `NMPEvalCap`, `NMPVerifyDepth`,
`NMPVerifyPercent`, `HindsightRecoverReduction`, `HindsightTrimReduction`,
`HindsightTrimEvalSum`, `CheckExtension`, `PromotionExtension`, `ExtensionPlyLimitPct`,
`HistoryBonusPerDepth`, `HistoryBonusMax`, `HistoryMalusPercent`.

Existing LMR option names are unchanged; `LMRHistoryCoeff` / `LMRContinuationCoeff` ranges widened to +/-1024.

## Tuning harness note

`tuning/spsa_lmr*.py` scored decisive games incorrectly before this change (mate was
printed as `cp`, and a mated root printed no mate score). Any earlier SPSA results should be treated as
suspect; re-run them on this build.

## Known remaining limitations (next phases)

- Movegen is still fully legal with make/undo per move; qsearch still generates quiet checks at every node.
- No futility/razoring/LMP/SEE pruning, no aspiration windows, no TT-stored eval.
- UCI is single-threaded: `stop`, `ponder` and a truly infinite `go` are not supported yet.
- Eval is still untapered.
