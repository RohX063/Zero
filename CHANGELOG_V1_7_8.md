# ZERO V1.7.8 Changelog

## Added
- `bitboard.h/.cpp`
- Primary occupancy bitboards in `Position`
- 0..63 square based `Move`
- Bitboard-based pseudo-legal move generation

## Changed
- `Position` no longer stores an 8x8 board.
- UCI/PST/move ordering consume square-based moves.
- FEN and make/undo now update bitboards incrementally.

## Intentionally unchanged
- Search algorithm
- Evaluation terms and values
- MovePicker scoring policy
- Thread/Worker architecture
- QSearch/search heuristics
