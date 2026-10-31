# Design: ZERO quiescence search

Status: DRAFT v0.2. Layer 0 step 1 implemented and proven. Layer 1 and the threat-aware layer are not started.
Authors: ZERO engineers (Rohan Singh and Claude), design by discussion and measurement.

## 1. Problem

At the search horizon, turn an unsettled position into a trustworthy score for the least work.
Measured on R3.4:

- Depth 3 in about 0.5 s in check-heavy positions (PGN of the 33-move game).
- Evaluation is about half of runtime; every qsearch node calls it for stand-pat.
- About 8 make/undo calls per evaluated node. Most of them were wasted (finding F1 below).
- No TT probe in qsearch.

## 2. Decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Speed first, threat-aware layer second | Measured speed problem; sharp ideas need a fast base |
| D2 | No aggressive pruning | Early experience with it was frustrating; identity is brilliancy and clean play |
| D3 | Organise by RISK, not by qsearch ply | Makes D2 enforceable and testable |
| D4 | Quiet checks are an explicit policy (`QSQuietChecks` 0/1/2), default 0 | R3.4's real behaviour was 0; intended behaviour must be measured, not assumed |

## 3. Organisation: layers ordered by risk

### Layer 0: Exact (same tree, same scores, only faster)
Acceptance: fixed-depth `bench` gives IDENTICAL best move, score and node count before and after.
- DONE: quiet-check test without make/undo (bitboard legality plus direct and discovered check).
- Todo: capture and promotion generation without full legal generation (pseudo-legal plus pin-aware legality).
- Todo: cheaper capture ordering; one static evaluation per node.

### Layer 1: Careful (changes the tree, with a measured risk budget)
- Quiet checks: off, first qsearch ply only, or a budgeted version (decide from E4 plus matches).
- TT probe and store in qsearch (stores static eval too).
- Skip clearly losing captures with a WIDE margin, never when the move gives check.
- Delta-style skip only with a wide margin.

### Layer 2: Aggressive (NOT in scope, decision D2)

### Layer 3: Threat-aware (after Layers 0 and 1 are stable)
Hypothesis: cheap danger signals (major piece en prise, heavy attack on our king zone) decide where extra
effort goes. Known tension: ZERO's offers (for example Ba3, rook left en prise) are proved by the main search;
a pessimistic qsearch could discourage them. The sacrifice gate decides.

## 4. Risk budget and shadow mode (our own rule for D2)

Every pruning rule gets a false-prune budget. In shadow mode (debug build) the rule says "skip" but the move is
still searched; whenever that move would have raised alpha it is counted as a false prune. A rule ships only if its
measured rate is at or below budget. First proposed budget: 0.1 % (to be confirmed). Margins come from ZERO's own
evaluation error (stand-pat versus deeper score), not from other engines.

## 5. Gates

| Gate | Pass condition |
|---|---|
| Layer 0 equivalence | identical move, score and node count on `bench` |
| Speed | nodes per second and depth at fixed time clearly better than R3.4 |
| Sacrifice gate | replay of the 277 and 104 sets not below the R3.4 baseline by more than noise |
| Strength | no loss against the frozen R3.4 over 400 games at 10+0.1, same EPD suite |
| Risk budget | each Layer 1 rule at or below its false-prune budget in shadow mode |

## 6. Progress log

- Layer 0 step 1 DONE. `QSQuietChecks 0` reproduces R3.4 exactly: identical best move, score and node count on all 101
  bench positions at depth 7 (14,265,862 nodes, signature 499f672cf52e32bc). Speed: +24 % and +31 % nodes per second
  in two runs (398k to 495k, then 344k to 452k; machine noise explains the spread).
- Equivalence test: bitboard generator versus make/undo oracle, 21,783 quiet checks over 18,407 positions, 0 mismatches.
- E4 (replay of 381 sacrifice positions, 300 ms, sandbox, measured twice):

| QSQuietChecks | 277-set | 104-set | avg depth |
|---|---|---|---|
| 0 (off, = R3.4) | 227 (82 %) | 86 (83 %) | 7.3 / 15.5 |
| 1 (first qsearch ply) | 212 (77 %) | 81 (78 %) | 6.9 / 14.6 |
| 2 (every ply) | 71 (26 %) | 26 (25 %) | 0.9 / 0.6 |

  Caveat: the positions come from games played with checks effectively off, so mode 0 is favoured by construction.
  Verdict so far: mode 2 is unusable (unbounded check chains); mode 1 costs about half a ply and shows no gain.
  Real checks need a hard budget before they deserve a strength match.

## 7. Open questions

- Q1 Confirm the 0.1 % false-prune budget, or choose another.
- Q2 Quiet checks: keep the policy at 0, or design a budgeted version (for example first ply and only when the enemy king
  zone is attacked)?
- Q3 Threat-aware layer gets its own page after Layer 1 is stable.

## 8. Provenance

Layers 0 and 1 use standard, widely published techniques. The risk-budget method, shadow mode and the threat-aware
layer are ZERO's own designs. Every parameter lands in docs/PARAMETER_PROVENANCE.md with date and evidence.

## 9. Findings log

- F1: quiet-check generation in R3.4 was dead code. A `Color` was passed where `Position::isKingInCheck` expects a bool
  (`true` = white king), so `legal` and `givesCheck` asked about the same king and could never both be true. The list was
  always empty, yet every qsearch node did a make/undo for every quiet move. Consequences:
  (a) an earlier claim "quiet checks do not produce the sacrifices" is RETRACTED: that experiment compared "no checks"
      with "no checks";
  (b) removing the dead work is an exact speed-up;
  (c) the intended behaviour (checks at every qsearch ply) explodes the tree (E4, mode 2);
  (d) lesson: an equivalence test must fail if its oracle never produces the thing under test. `quiet_checks_test` does.
