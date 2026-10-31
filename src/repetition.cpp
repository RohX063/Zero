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

bool isInsufficientMaterial(const Position& position)
{
    using namespace Zero;
    if (position.pieces(PAWN) | position.pieces(ROOK) | position.pieces(QUEEN))
        return false;

    const int minors = popcount(position.pieces(KNIGHT) | position.pieces(BISHOP));
    return minors <= 1;
}

bool isDrawInSearch(const Position& position, int ply)
{
    const StateInfo* current = position.state();
    if (current == nullptr)
        return false;

    if (position.halfmove_clock() >= 100
        && !position.isKingInCheck(position.isWhiteToMove()))
        return true;

    if (isInsufficientMaterial(position))
        return true;

    // Null-move states and freshly irreversible positions cannot repeat.
    if (current->isNullMove || current->halfmoveClock == 0)
        return false;

    const Key key = current->key;
    const int reach = current->halfmoveClock;
    int distance = 0;
    int earlier = 0;

    for (const StateInfo* state = current->previous;
         state != nullptr && distance < reach;
         state = state->previous)
    {
        ++distance;
        if (state->isNullMove)
            break;

        // Same side to move requires an even distance.
        if ((distance & 1) == 0 && state->key == key) {
            if (distance < ply)
                return true;           // repetition inside the search tree
            if (++earlier >= 2)
                return true;           // third occurrence including history
        }
    }

    return false;
}

bool moveCausesThreefold(Position& position, const Move& move)
{
    // A third occurrence is only reachable if the current position has
    // already appeared at least twice. Avoid a speculative make/undo in the
    // overwhelmingly common non-repetition case.
    if (repetitionCount(position) < 2)
        return false;

    StateInfo nextState;
    position.doMove(move, nextState);
    const bool repeats = isThreefoldRepetition(position);
    position.undoMove(move);
    return repeats;
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
