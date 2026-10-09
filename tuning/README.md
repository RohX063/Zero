# ZERO LMR SPSA Tuner

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## One manual candidate

```bash
./build/zero
setoption name LMRBase value 982
setoption name LMRDepthCoeff value 0
setoption name LMRMoveCoeff value 0
```

## SPSA smoke test

Run a tiny experiment first:

```bash
python3 tuning/spsa_lmr.py \
  --engine ./build/zero \
  --iterations 2 \
  --games 2 \
  --movetime 50 \
  --positions tuning/regression.fen
```

For real tuning, increase the games substantially and use a repeatable hardware/time-control setup. Search-parameter tuning should use normal time controls for realistic scaling; fixed-depth/low-time runs are best treated as fast smoke tests.

## What the tuner protects against

- candidate-vs-candidate SPSA signal uses matched openings and swapped colors,
- an anchor regression gate protects known positions,
- node efficiency is recorded but is not allowed to dominate the chess result,
- state is resumable from JSON,
- the holdout path should be kept separate from tuning once a stable candidate emerges.


## Phase-1 game-result safeguard

Normal unfinished plies must not be counted as wins or losses. Exact mate/stalemate results are used whenever available; fast games reaching the ply cap receive a bounded score-based adjudication solely to avoid a flat all-draw SPSA signal. This soft signal is not an Elo estimate and must be validated by an independent holdout/gauntlet.
