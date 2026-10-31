# Sacrifice suite

Positions (side to move = ZERO) involving a sacrifice. EPD operations:
`bm` engine-best move (only once confirmed by a strong reference), `sac` the sacrifice played, `uci` the move in
long notation, `c0` source and evidence.

## Files

| File | Content |
|---|---|
| `sacrifice_v17.epd` | Two positions from a V1.7 game, from screenshots |
| `sacrifice_selfplay_clean.epd` | Three moves labelled Brilliant in the R3.4-vs-R3.3 match (rounds 2 and 3) |
| `sacrifice_candidates_277.epd` | 277 machine-mined material sacrifices (still down four plies later, engine eval at least +0.30) |
| `sacrifice_offers_major.epd` | 104 declined offers: a rook or queen left en prise, engine eval at least -0.30 |

## Findings

- The review site's "Brilliant" label is not the same as "material sacrifice": Nxc5 is a sacrifice, Bc4 a deflection,
  and Ba3 an offered rook (the h1 rook was attacked by the g3 knight, Black declined).
- Stockfish (local, depth 24 to 30): position 1 of V1.7 best move Be4 (5th-best still +7.06), position 2 best move Nxg6.
  So the suite needs two kinds: Type A (the sacrifice is the best move) and Type B (one of several winning moves).
- Replay at 300 ms (plays the original move again), sandbox:

| Set | R3.4 behaviour (QSQuietChecks 0) |
|---|---|
| 277 candidates | 227 (82 %) |
| 104 major offers | 86 (83 %) |

## RETRACTION (quiet checks)

An earlier version of this file said quiet checks in qsearch do not produce the sacrifices. That is WITHDRAWN.
R3.4's quiet-check generator never produced a move (dead code), so the "with" and "without" runs were the same search.
The real ablation (modes 0, 1, 2) is in `design/qsearch.md` section 6. Replay rates are biased towards mode 0 because the
positions come from games played in that mode, so they are a regression gate, not proof that checks are useless.
