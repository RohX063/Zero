#include "repetition.h"

#include "position.h"

namespace Zero::Rules {

int repetitionCount(const Position& position)
{
    const StateInfo* current = position.state();

    if (current == nullptr || current->isNullMove)
        return 0;

    const Key key = current->key;
    int count = 1;

    // Only reversible history can possibly contain the same position. Once a
    // pawn move or capture reset the halfmove clock to zero, no older position
    // can be identical to the current material configuration.
    for (const StateInfo* state = current->previous;
         state != nullptr;
         state = state->previous)
    {
        if (state->isNullMove)
            break;

        if (state->key == key)
            ++count;

        if (state->halfmoveClock == 0)
            break;
    }

    return count;
}

bool isThreefoldRepetition(const Position& position)
{
    return repetitionCount(position) >= 3;
}

bool isFiftyMoveClaimable(const Position& position)
{
    return position.halfmove_clock() >= 100;
}

bool isSeventyFiveMoveRule(const Position& position)
{
    return position.halfmove_clock() >= 150;
}

bool isAutomaticDraw(const Position& position)
{
    return isThreefoldRepetition(position)
        || isSeventyFiveMoveRule(position);
}

} // namespace Zero::Rules
