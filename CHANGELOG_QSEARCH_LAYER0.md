# Qsearch Layer 0 (exact speed-up), engine id "ZERO 2.6 R3.5 QS-L0"

## Finding F1: dead quiet-check generator
`generateQuietChecks` passed a `Color` to `Position::isKingInCheck(bool whiteKing)`. `WHITE` converts to false and
`BLACK` to true, so `legal` and `givesCheck` asked about the same king and could never both be true. The list was
always empty, yet every qsearch node did a make/undo (two king-attack tests) for every quiet move.

## Changes
- `generateQuietChecks` uses an exact bitboard test (no make/undo): legality after the move (including the castling rook
  and x-ray through the king) and direct or discovered checks. Same pseudo-move order as before.
- New policy `QSQuietChecks` (0 off, 1 first qsearch ply, 2 every ply). Default 0 = R3.4's real behaviour.
- New `bench [depth] [file]` UCI command: deterministic fixed-depth run, one line per position and a signature.
- New test `quiet_checks_test`: generator versus a make/undo oracle (21,783 checks, 18,407 positions, 0 mismatches);
  it fails if the oracle sees no checks at all.
- `tuning/bench_positions.epd`: 101 positions. `tuning/tools/pgnpos.cpp`: PGN replay and sacrifice miner.
- `design/`: qsearch design page and the sacrifice suites.

## Result (QSQuietChecks 0)
Identical best move, score and node count on all 101 bench positions at depth 7 (14,265,862 nodes,
signature 499f672cf52e32bc). Nodes per second: +24 % (398k to 495k) and +31 % (344k to 452k) in two runs.
