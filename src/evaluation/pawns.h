#ifndef ZERO_EVALUATION_PAWNS_H
#define ZERO_EVALUATION_PAWNS_H

#include "../bitboard.h"
#include "../types.h"

class Position;

namespace Zero::Evaluation {

struct PawnFeatures {
    int pawns = 0;
    int doubled = 0;
    int isolated = 0;
    int backward = 0;
    int passed = 0;
    int connected = 0;
    int islands = 0;
};

// Analyze one side's pawn structure from the given friendly/enemy pawn bitboards.
PawnFeatures analyzePawnStructure(Bitboard friendlyPawns, Bitboard enemyPawns, Color color);

// Pawn-structure evaluation from White's point of view.
int evaluatePawns(const Position& position);

} // namespace Zero::Evaluation

#endif
