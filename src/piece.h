#ifndef ZERO_PIECE_H
#define ZERO_PIECE_H

#include "types.h"

enum Piece : uint8_t
{
    EMPTY = 0,
    WHITE_PAWN,
    WHITE_KNIGHT,
    WHITE_BISHOP,
    WHITE_ROOK,
    WHITE_QUEEN,
    WHITE_KING,
    BLACK_PAWN,
    BLACK_KNIGHT,
    BLACK_BISHOP,
    BLACK_ROOK,
    BLACK_QUEEN,
    BLACK_KING,
    PIECE_NB
};

inline constexpr bool isWhitePiece(Piece piece)
{
    return piece >= WHITE_PAWN && piece <= WHITE_KING;
}

inline constexpr bool isBlackPiece(Piece piece)
{
    return piece >= BLACK_PAWN && piece <= BLACK_KING;
}

inline constexpr Zero::Color color_of(Piece piece)
{
    return isBlackPiece(piece) ? Zero::BLACK : Zero::WHITE;
}

inline constexpr Zero::PieceType type_of(Piece piece)
{
    if (piece == EMPTY) return Zero::NO_PIECE_TYPE;
    const int n = int(piece <= WHITE_KING ? piece : piece - BLACK_PAWN + 1);
    return Zero::PieceType(n);
}

inline constexpr Piece make_piece(Zero::Color c, Zero::PieceType pt)
{
    if (pt == Zero::NO_PIECE_TYPE) return EMPTY;
    const Piece base = c == Zero::WHITE ? WHITE_PAWN : BLACK_PAWN;
    return Piece(base + int(pt) - 1);
}

#endif
