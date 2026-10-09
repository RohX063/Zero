# ZERO R3 Industrial Search Audit

This checkpoint focuses on correctness and search architecture. Stockfish 19 (tag `sf_19`) was used only as an architectural reference, not as a source for copied implementation.

## Correctness fixes

- Corrected en-passant target validation in FEN loading.
- Corrected en-passant target creation after double pawn pushes.
- Made move identity semantic (`from`, `to`, promotion choice) so TT/killer/counter moves remain comparable to freshly generated special moves.
- Prevented TT occupancy from depending solely on a non-zero Zobrist key.
- Fixed CMake source completeness so all linked implementation units are actually compiled.
- Prevented reduced null-move searches from being recorded in the TT as full-depth proofs.
- Prevented fail-low nodes from advertising a non-improving move as their TT move.
- Restricted TT cutoffs to non-PV nodes while still using TT moves for ordering at PV nodes.
- Added mate-distance window clamping in the main search.
- Moved the main search to pseudo-legal generation with lazy legality validation; illegal moves no longer affect move count, history learning, or LMR pressure.
- Corrected the previous-move history context to use the move leading into the current node.
- Removed an unused evaluation helper so the release build is warning-clean under the project's current warning set.

## Search architecture upgrades

- MovePicker now uses explicit stages: TT, captures/promotions, special quiets, then quiets.
- Quiet history is used during staged selection rather than globally sorting every move up front.
- LMR now uses a smooth logarithmic base plus history/PV adjustments and protects TT, counter, killer, promotion, and checking moves.
- History bonuses are depth-sensitive and bounded.
- Check and promotion extensions are retained.
- Null-move pruning remains conservative and excludes PV nodes and obvious pawn/king-only zugzwang material.

## Validation

Release build:

- Build: PASS
- perft: PASS (all 6 canonical positions)
- search_smoke: PASS
- position_integrity: PASS
- Compiler warnings: none in the modified build

Debug + ASan/UBSan build:

- perft: PASS
- search_smoke: PASS
- position_integrity: PASS
- sanitizer errors: none observed

## Scope note

Passing the current regression and sanitizer suites does not mathematically prove ZERO is bug-free or that the search parameters are statistically optimal. The next engineering step is objective engine testing (tactical suites, fixed-position regression, and self-play/Elo measurement) before claiming a tuned strength gain.
