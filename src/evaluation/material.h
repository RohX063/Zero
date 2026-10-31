#ifndef ZERO_EVALUATION_MATERIAL_H
#define ZERO_EVALUATION_MATERIAL_H

#include "../piece.h"

class Position;

namespace Zero::Evaluation {

// Material balance from White's point of view, excluding kings.
int evaluateMaterial(const Position& position);

} // namespace Zero::Evaluation

// Kept as the engine-wide move-ordering API for compatibility.
int pieceValue(Piece piece);

#endif
