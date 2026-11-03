# Hindsight diagnostics (experimental, not engine changes)

## Purpose

Measure what the current mechanism actually does before proposing replacements. The engine at base commit `4087cbc70c5a7dcdc50f2047c43d290863c3fb2c` remains unchanged. All tests used local single-threaded GCC 12.2 `-O3 -DNDEBUG` builds, 16 MB hash, fixed-depth iterative search and a fresh worker/hash/history per position. These are small, deterministic search diagnostics, not playing-strength tests.

`positions.epd` is lines 121–144 of `tuning/zero_openings_8ply.epd`; `positions-deep.epd` is its first eight positions. The input represents opening positions, not middlegame/endgame coverage; a four-field EPD has no game history and the halfmove counter is reset to zero. The known baseline repetition/root and qsearch limitations remain present.

## Instrumentation results

The instrumentation adds counters only. On the 101-position depth-5 benchmark, instrumented and original engines both produced 3,408,585 nodes and signature `95212ceb231e1662`. This is a check against accidental behavior changes, not a proof for every position. Counts below aggregate all iterations through the requested depth.

| Metric | 24 positions, depth 7 | 8 positions, depth 9 |
|---|---:|---:|
| Visited nodes (includes qsearch) | 2,833,107 | 4,915,573 |
| Hindsight adjustment eligible calls | 8,302 | 15,796 |
| Effective depth increases | 0 | 831 |
| Effective depth decreases | 0 | 0 |
| Separate punishment hook reduced an LMR reduction | 219 | 200 |
| Of those, reduced it to zero | 150 | 152 |
| Eligible calls with prior reduction 0 / 1 / 2 / 3+ | 7,023 / 1,128 / 151 / 0 | 10,922 / 2,215 / 1,379 / 1,280 |

Eligibility is after the normal draw, TT and eval checks, and requires valid current and parent static evals, ply >= 2 and depth >= 3. `up` and `down` count **net** changes after all correction conditions and the maximum-depth clamp. `punished` counts cases in which the separate `punishOpponentWorsening` hook actually lowered the selected reduction. Nodes and events are correlated; counts are not independent observations.

The depth-7 absence of depth increases is consistent with there being no eligible prior reductions of 3 or more; it does **not** imply the mechanism never matters at other depths or positions. The fact that no depth-decreases occurred here does not prove the rule is unreachable. See `depth7.csv` and `depth9.csv` for per-position data.

## Matched fixed-depth ablations

Four builds searched the same eight depth-9 positions with the same reset semantics:

| Build | Nodes | Relative to original | Changed best moves / scores |
|---|---:|---:|---:|
| Original, both mechanisms | 4,915,573 | — | — |
| Depth adjustment removed only | 4,921,851 | +0.13% | 0 / 2 of 8 |
| Punishment hook removed only | 5,320,490 | +8.24% | 0 / 4 of 8 |
| Both removed | 4,951,065 | +0.72% | 0 / 4 of 8 |

These node counts reflect changes to move ordering and the selective search tree. Fewer nodes or matching best moves at one depth do **not** establish higher strength. Timing and NPS vary with hardware and were not used as an acceptance criterion. The interaction between components and changing search trees means differences are not additive. Per-position `*-bench9.txt` files contain scores, moves and nodes.

## Reproducing the diagnostic build

Use a disposable checkout at the base commit; do not apply patches to the working engine. Copy the diagnostics directory somewhere safe before checking out if needed. Apply `instrumentation.patch`; its only changes are counters in `search.cpp` and `search_types.h`. Compile the driver with all engine C++ files except `main.cpp` and `uci.cpp`:

```sh
git apply /path/to/instrumentation.patch
sources=$(find src -name '*.cpp' ! -name main.cpp ! -name uci.cpp)
g++ -std=c++17 -O3 -DNDEBUG -Isrc /path/to/measure.cpp $sources -o /tmp/zero-hindsight-measure
/tmp/zero-hindsight-measure /path/to/positions.epd 7
/tmp/zero-hindsight-measure /path/to/positions-deep.epd 9
```

For the matched search ablations, start from a fresh unmodified checkout for each variant and apply exactly one of `ablate-depth.patch`, `ablate-punish.patch`, or `../hindsight-ablation/remove-hindsight.patch`. Build each using the same flags. Feed `bench 9 /path/to/positions-deep.epd` to each engine, followed by `quit`. Do not combine the instrumentation and ablation patches without resolving overlaps.

## Next test design

The old system has two separable effects, so evaluate them factorially: both on, depth correction off, punishment off, and both off. First verify builds and run matched fixed-depth diagnostics on diverse opening, middlegame and endgame suites. For strength, predefine a game budget and acceptance rule, run paired colors on fresh openings with identical hash, hardware and time controls, and report the pair-level results and unfinished games. Confirm any positive screening result on a disjoint holdout and at a longer time control. Repeatedly testing new candidates on the same opening set risks overfitting; no small pilot is a superiority verdict.
