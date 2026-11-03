# Experimental screening: recapture extension

This candidate was tested only in an isolated temporary source tree and was **not promoted or retained as an engine change**. The idea came from the general discussion at https://www.chessprogramming.org/Recapture_Extensions. The experiment started from the no-hindsight control, then extended a recapture by one ply only when: the previous move captured, the current move captures on the same destination, the node is a principal-variation node, depth is at most four, the existing extension is zero, and ZERO's existing root-relative extension cap allows it. This is a selectivity/move-ordering experiment, not a direct implementation of the old hindsight mechanism.

Seven Debug tests passed, including policy boundary tests. A start-position depth-7 diagnostic executed 11 recapture extensions and confirmed that the root position was restored. This is not exhaustive correctness testing.

## Screening match

- Baseline: repository commit `4087cbc70c5a7dcdc50f2047c43d290863c3fb2c`, original hindsight enabled. Both engines built with `g++ -std=c++17 -O3 -DNDEBUG`, 16 MB hash and one thread.
- 24 positions from `tuning/zero_openings_8ply.epd` (zero-based lines 96–119), each played twice with colors swapped, 30 ms per move. Python-chess handled legal moves, draws and game termination; the cap was 240 plies, with no score adjudication.
- Candidate result: 16 wins, 13 draws, 18 losses, one unfinished game. Completed-game score: 22.5/47 = 47.9%. This is a small, noisy screen, not an Elo estimate.
- `screening.pgn` and `screening.log` record the games. Baseline binary SHA-256: `2b4dac9bb7b7d61e604187a225375be79af999eb45bde84a23ffa77b62d1a207`; candidate binary SHA-256: `62c65ae7270526b1f531a74c7a8d859b1a10be5b47516a31c94fa3250097da0e`.

No superiority was demonstrated. The candidate itself was not saved; preserve these results as a screening record, not as a reproducible source artifact. The original engine has known correctness and protocol limitations, which both sides share.
