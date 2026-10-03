#ifndef ZERO_REPETITION_H
#define ZERO_REPETITION_H

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

} // namespace Zero::Rules

#endif
