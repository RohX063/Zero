#ifndef ZERO_EVALUATION_MOBILITY_H
#define ZERO_EVALUATION_MOBILITY_H

#include "../bitboard.h"
#include "../types.h"

class Position;

namespace Zero::Evaluation {

// Mobility measurements for one side. Kings and pawns are intentionally
// excluded here; king mobility belongs to king safety and pawn movement to
// pawn structure.
struct MobilityFeatures {
    int knight = 0;
    int bishop = 0;
    int rook = 0;
    int queen = 0;

    int safeKnight = 0;
    int safeBishop = 0;
    int safeRook = 0;
    int safeQueen = 0;
};

// Analyze non-pawn, non-king piece mobility for one side.
MobilityFeatures analyzeMobility(const Position& position, Color color);

// Mobility evaluation from White's point of view.
int evaluateMobility(const Position& position);

} // namespace Zero::Evaluation

#endif
