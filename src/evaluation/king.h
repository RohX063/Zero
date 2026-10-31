#ifndef ZERO_EVALUATION_KING_H
#define ZERO_EVALUATION_KING_H

#include "../bitboard.h"
#include "../types.h"

class Position;

namespace Zero::Evaluation {

struct KingSafetyFeatures {
    Square kingSquare = SQ_NONE;
    int shieldPawns = 0;
    int shieldStrength = 0;
    int missingShield = 0;
    int pawnStorm = 0;
    int ringAttackedSquares = 0;
    int ringAttackers = 0;
    int directAttackers = 0;
    int pawnRingAttacks = 0;
    int heavyRingAttackers = 0;
    int xrayPressure = 0;
    int safeEscapeSquares = 0;
    int unsafeEscapeSquares = 0;
    int phase256 = 0;
    int score = 0;
};

KingSafetyFeatures analyzeKingSafety(const Position& position, Color color);
int evaluateKingSafety(const Position& position);

} // namespace Zero::Evaluation

#endif
