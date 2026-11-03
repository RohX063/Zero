# Experimental control: remove hindsight without replacement

This is a preserved **experiment**, not the active engine and not a proven improvement. The patch applies to repository commit `4087cbc70c5a7dcdc50f2047c43d290863c3fb2c`:

```sh
git apply --check experiments/hindsight-ablation/remove-hindsight.patch
# Apply only in a separate, disposable checkout/worktree for testing:
git apply /path/to/remove-hindsight.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The patch removes the hindsight depth correction, the opponent-worsening reduction adjustment, their UCI options and the old mechanism-specific test. It does not replace them. The six remaining Debug tests pass. Production source in the main checkout is unchanged.

## Screening match

- Original baseline versus no-hindsight control, `g++ -std=c++17 -O3 -DNDEBUG`, 16 MB hash, one thread.
- 24 positions from `tuning/zero_openings_8ply.epd` (zero-based lines 48–71), each played twice with colours swapped; FEN counters set to 0 and 1.
- Fixed 30 ms per move, python-chess rules and draw-claim handling, 240-ply cap. No score-based adjudication. One capped game was unfinished and remains `*` in the PGN.
- Control result: 16 wins, 18 draws, 13 losses, one unfinished. Among 47 finished games the score is 53.2%; this is only a small noisy screen, not evidence of superiority.
- `screening.pgn` contains the games and `screening.log` the game-by-game summary. Original binary SHA-256: `2b4dac9bb7b7d61e604187a225375be79af999eb45bde84a23ffa77b62d1a207`. Tested control binary SHA-256: `3e51c9e726307ddb9792ee4477f1de444ff557fbadf45dc3ed63e7e9c88436b4`.

The recorded match used a control build that still contained an unused experimental helper and unit test, neither referenced by the engine search. This patch removes that dead scaffolding. The cleaned control and the match binary produced the same depth-5 benchmark signature `bdf7eaf4a830577e` and 3,400,997 nodes over `tuning/bench_positions.epd`. This is a useful equivalence check, not proof that their behavior is identical in every position.

The baseline has known search and draw-handling limitations; both sides share them. Use a larger independent matched-opening test before drawing conclusions. Do not apply this patch to the main engine or push it as a strength improvement without a separate decision.
