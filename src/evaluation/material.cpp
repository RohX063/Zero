#include "material.h"

#include "../position.h"

namespace Zero::Evaluation {

namespace {

constexpr int PAWN_VALUE   = 100;
constexpr int KNIGHT_VALUE = 320;
constexpr int BISHOP_VALUE = 330;
constexpr int ROOK_VALUE   = 500;
constexpr int QUEEN_VALUE  = 900;

} // namespace

int evaluateMaterial(const Position& position)
{
    using namespace Zero;

    const auto white = [&](PieceType type) {
        return popcount(position.pieces(WHITE, type));
    };
    const auto black = [&](PieceType type) {
        return popcount(position.pieces(BLACK, type));
    };

    return PAWN_VALUE   * (white(PAWN)   - black(PAWN))
         + KNIGHT_VALUE * (white(KNIGHT) - black(KNIGHT))
         + BISHOP_VALUE * (white(BISHOP) - black(BISHOP))
         + ROOK_VALUE   * (white(ROOK)   - black(ROOK))
         + QUEEN_VALUE  * (white(QUEEN)  - black(QUEEN));
}

} // namespace Zero::Evaluation

int pieceValue(Piece piece)
{
    switch (piece) {
        case WHITE_PAWN: case BLACK_PAWN: return 100;
        case WHITE_KNIGHT: case BLACK_KNIGHT: return 320;
        case WHITE_BISHOP: case BLACK_BISHOP: return 330;
        case WHITE_ROOK: case BLACK_ROOK: return 500;
        case WHITE_QUEEN: case BLACK_QUEEN: return 900;
        case WHITE_KING: case BLACK_KING: return 20000;
        default: return 0;
    }
}
