#ifndef ZERO_BITBOARD_H
#define ZERO_BITBOARD_H

#include "types.h"

namespace Zero::Bitboards {

void init();

extern Bitboard KnightAttacks[SQUARE_NB];
extern Bitboard KingAttacks[SQUARE_NB];
extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];

extern const Bitboard FileBB[8];
extern const Bitboard RankBB[8];

Bitboard bishop_attacks(Square sq, Bitboard occupied);
Bitboard rook_attacks(Square sq, Bitboard occupied);
Bitboard queen_attacks(Square sq, Bitboard occupied);

}

#endif
