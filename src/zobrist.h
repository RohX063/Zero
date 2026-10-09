#ifndef ZERO_ZOBRIST_H
#define ZERO_ZOBRIST_H

#include "piece.h"
#include "types.h"

namespace Zero::Zobrist {

extern Bitboard psq[PIECE_NB][SQUARE_NB];
extern Key side;
extern Key castling[16];
extern Key enPassantFile[8];

void init();

inline Key piece(Piece pc, Square sq) { return psq[pc][sq]; }
inline Key castling_key(int rights) { return castling[rights & 15]; }
inline Key en_passant_key(Square sq) { return sq < SQUARE_NB ? enPassantFile[file_of(sq)] : 0; }

} // namespace Zero::Zobrist

#endif
