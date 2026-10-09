#ifndef ZERO_SEARCH_STACK_H
#define ZERO_SEARCH_STACK_H

#include "move.h"
#include "piece.h"
#include "types.h"

namespace Zero::Search {

// Per-ply search context. Kept independent from Worker so node state can evolve
// without expanding the public Worker interface.
struct Stack {
    Move currentMove{};
    Move excludedMove{};
    Piece movedPiece = EMPTY;
    Value staticEval = 0;
    int ply = 0;
    int moveCount = 0;
    int nmpMinPly = 0;
    int reduction = 0;
    bool inCheck = false;
    // True only when staticEval holds a real evaluation for this node (nodes in
    // check, shallow nodes and fresh stack slots do not).
    bool evalValid = false;
    bool canNullMove = true;
    bool isNullMove = false;
};

} // namespace Zero::Search

#endif
