#ifndef ZERO_REPETITION_H
#define ZERO_REPETITION_H

#include "move.h"

class Position;

namespace Zero::Rules {

// Number of occurrences of the current position in the available state chain,
// including the current position itself. Null-move states are excluded.
int repetitionCount(const Position& position);

// Search-safe repetition draw recognition.
bool isThreefoldRepetition(const Position& position);

// Rule counters exposed separately so the search and UCI layers can choose
// claimable versus automatic draw semantics deliberately.
bool isFiftyMoveClaimable(const Position& position);
bool isSeventyFiveMoveRule(const Position& position);

// Automatic draw conditions suitable for the search core.
bool isAutomaticDraw(const Position& position);

// True when neither side can possibly deliver mate (K vs K, or K + one minor
// piece vs K). Deliberately conservative: other drawn endings are left to eval.
bool isInsufficientMaterial(const Position& position);

// Draw test for nodes inside the search tree. `ply` is the distance from the
// search root. A position is scored as a draw when:
//   * it repeats an earlier position that occurred AFTER the root (one
//     repetition inside the tree is enough: the side that can repeat can force it),
//   * it has occurred twice before (true threefold including game history),
//   * the fifty-move counter reached 100 and the side to move is not in check,
//   * there is insufficient mating material.
bool isDrawInSearch(const Position& position, int ply);

// True when making the supplied legal move would create the third occurrence
// of the resulting position in the current state chain. The move is applied
// and undone internally, so callers keep ownership of the real search state.
bool moveCausesThreefold(Position& position, const Move& move);

} // namespace Zero::Rules

#endif
