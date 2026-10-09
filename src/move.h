#ifndef ZERO_MOVE_H
#define ZERO_MOVE_H

#include "piece.h"
#include "types.h"

struct Move
{
    Zero::Square from = Zero::SQ_NONE;
    Zero::Square to   = Zero::SQ_NONE;

    Piece promotionPiece = EMPTY;
    uint8_t flags = 0;

    enum Flag : uint8_t {
        NONE            = 0,
        PROMOTION       = 1 << 0,
        EN_PASSANT      = 1 << 1,
        CASTLE_KING     = 1 << 2,
        CASTLE_QUEEN    = 1 << 3
    };

    static Move none() { return {}; }

    bool isOk() const { return from < Zero::SQUARE_NB && to < Zero::SQUARE_NB; }
    bool isPromotion() const { return flags & PROMOTION; }
    bool isEnPassant() const { return flags & EN_PASSANT; }
    bool isCastle() const { return flags & (CASTLE_KING | CASTLE_QUEEN); }
    bool isKingSideCastle() const { return flags & CASTLE_KING; }
    bool isQueenSideCastle() const { return flags & CASTLE_QUEEN; }

    Zero::Square from_sq() const { return from; }
    Zero::Square to_sq() const { return to; }
};

// Move identity is determined by from/to and promotion choice.
// Castling and en-passant flags are position-dependent details reconstructed
// from the legal move list; excluding them keeps TT/killer/counter moves
// comparable to freshly generated moves.
inline bool operator==(const Move& a, const Move& b)
{
    return a.from == b.from
        && a.to == b.to
        && a.promotionPiece == b.promotionPiece;
}

inline bool operator!=(const Move& a, const Move& b) { return !(a == b); }

#endif
