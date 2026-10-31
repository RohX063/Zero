#include "mobility.h"

#include "../position.h"
#include "../piece.h"

#include <algorithm>

namespace Zero::Evaluation {

namespace {

constexpr int KNIGHT_MOBILITY[9] = {
    -30, -22, -14, -7, 0, 7, 14, 22, 30
};

constexpr int BISHOP_MOBILITY[14] = {
    -24, -20, -16, -12, -8, -4, 0, 4,
     8,  12,  16,  20, 24, 28
};

constexpr int ROOK_MOBILITY[15] = {
    -20, -17, -14, -11, -8, -5, -2, 1,
      4,   7,  10,  13, 16, 19, 22
};

constexpr int QUEEN_MOBILITY[28] = {
    -18, -16, -14, -12, -10, -8, -6, -4,
     -2,   0,   2,   4,   6,  8, 10, 12,
     14,  16,  18, 20, 22, 24, 26, 28,
     30,  32,  34, 36
};

constexpr int SAFE_MOBILITY_SCALE = 2;

int mobilityBonus(PieceType type, int mobility)
{
    switch (type) {
        case KNIGHT: return KNIGHT_MOBILITY[mobility];
        case BISHOP: return BISHOP_MOBILITY[mobility];
        case ROOK:   return ROOK_MOBILITY[mobility];
        case QUEEN:  return QUEEN_MOBILITY[mobility];
        default:     return 0;
    }
}

Bitboard pawnAttackMap(const Position& position, Color by)
{
    Bitboard attacks = 0;
    Bitboard pawns = position.pieces(by, PAWN);

    while (pawns) {
        const Square sq = pop_lsb(pawns);
        attacks |= Bitboards::PawnAttacks[by][sq];
    }

    return attacks;
}

} // namespace

MobilityFeatures analyzeMobility(const Position& position, Color color)
{
    using namespace Zero;

    MobilityFeatures result;
    const Color enemy = ~color;
    const Bitboard occupied = position.pieces();
    const Bitboard ownPieces = position.pieces(color);
    const Bitboard enemyKing = position.pieces(enemy, KING);
    const Bitboard enemyPawnAttacks = pawnAttackMap(position, enemy);

    Bitboard knights = position.pieces(color, KNIGHT);
    while (knights) {
        const Square sq = pop_lsb(knights);
        const Bitboard attacks = Bitboards::KnightAttacks[sq];
        const Bitboard destinations = attacks & ~ownPieces & ~enemyKing;
        result.knight += popcount(destinations);
        result.safeKnight += popcount(destinations & ~enemyPawnAttacks);
    }

    Bitboard bishops = position.pieces(color, BISHOP);
    while (bishops) {
        const Square sq = pop_lsb(bishops);
        const Bitboard attacks = Bitboards::bishop_attacks(sq, occupied);
        const Bitboard destinations = attacks & ~ownPieces & ~enemyKing;
        result.bishop += popcount(destinations);
        result.safeBishop += popcount(destinations & ~enemyPawnAttacks);
    }

    Bitboard rooks = position.pieces(color, ROOK);
    while (rooks) {
        const Square sq = pop_lsb(rooks);
        const Bitboard attacks = Bitboards::rook_attacks(sq, occupied);
        const Bitboard destinations = attacks & ~ownPieces & ~enemyKing;
        result.rook += popcount(destinations);
        result.safeRook += popcount(destinations & ~enemyPawnAttacks);
    }

    Bitboard queens = position.pieces(color, QUEEN);
    while (queens) {
        const Square sq = pop_lsb(queens);
        const Bitboard attacks = Bitboards::queen_attacks(sq, occupied);
        const Bitboard destinations = attacks & ~ownPieces & ~enemyKing;
        result.queen += popcount(destinations);
        result.safeQueen += popcount(destinations & ~enemyPawnAttacks);
    }

    return result;
}

int evaluateSideMobility(const Position& position, Color color)
{
    const Color enemy = ~color;
    const Bitboard occupied = position.pieces();
    const Bitboard ownPieces = position.pieces(color);
    const Bitboard enemyKing = position.pieces(enemy, KING);
    const Bitboard enemyPawnAttacks = pawnAttackMap(position, enemy);

    int score = 0;

    auto addPiece = [&](PieceType type, Bitboard attacks) {
        const Bitboard destinations = attacks & ~ownPieces & ~enemyKing;
        const int mobility = popcount(destinations);
        const int safe = popcount(destinations & ~enemyPawnAttacks);

        score += mobilityBonus(type, mobility);

        // Reward positive safe-space access, but do not turn an unsafe attack
        // map into an outsized negative penalty. The evaluator remains stable
        // when a piece has many legal but tactically sensitive squares.
        const int safeSurplus = std::max(0, safe - mobility / 2);
        score += SAFE_MOBILITY_SCALE * std::min(6, safeSurplus);
    };

    Bitboard pieces = position.pieces(color, KNIGHT);
    while (pieces) {
        const Square sq = pop_lsb(pieces);
        addPiece(KNIGHT, Bitboards::KnightAttacks[sq]);
    }

    pieces = position.pieces(color, BISHOP);
    while (pieces) {
        const Square sq = pop_lsb(pieces);
        addPiece(BISHOP, Bitboards::bishop_attacks(sq, occupied));
    }

    pieces = position.pieces(color, ROOK);
    while (pieces) {
        const Square sq = pop_lsb(pieces);
        addPiece(ROOK, Bitboards::rook_attacks(sq, occupied));
    }

    pieces = position.pieces(color, QUEEN);
    while (pieces) {
        const Square sq = pop_lsb(pieces);
        addPiece(QUEEN, Bitboards::queen_attacks(sq, occupied));
    }

    return score;
}

int evaluateMobility(const Position& position)
{
    return evaluateSideMobility(position, WHITE)
         - evaluateSideMobility(position, BLACK);
}

} // namespace Zero::Evaluation
