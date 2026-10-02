#include "mobility.h"

#include "../position.h"
#include "../piece.h"

namespace Zero::Evaluation {

namespace {

// The tables are centered around normal useful mobility rather than making
// every additional square worth the same amount. This keeps a queen with a
// huge attack map from dominating the entire evaluator.
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

// Queen mobility is deliberately compressed: queen activity is important,
// but raw move count must not overwhelm material, pawn structure or king safety.
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
        result.knight += popcount(attacks & ~ownPieces & ~enemyKing);

        Bitboard safe = attacks & ~ownPieces & ~enemyKing & ~enemyPawnAttacks;
        result.safeKnight += popcount(safe);
    }

    Bitboard bishops = position.pieces(color, BISHOP);
    while (bishops) {
        const Square sq = pop_lsb(bishops);
        const Bitboard attacks = Bitboards::bishop_attacks(sq, occupied);
        result.bishop += popcount(attacks & ~ownPieces & ~enemyKing);
        result.safeBishop += popcount(attacks & ~ownPieces & ~enemyKing & ~enemyPawnAttacks);
    }

    Bitboard rooks = position.pieces(color, ROOK);
    while (rooks) {
        const Square sq = pop_lsb(rooks);
        const Bitboard attacks = Bitboards::rook_attacks(sq, occupied);
        result.rook += popcount(attacks & ~ownPieces & ~enemyKing);
        result.safeRook += popcount(attacks & ~ownPieces & ~enemyKing & ~enemyPawnAttacks);
    }

    Bitboard queens = position.pieces(color, QUEEN);
    while (queens) {
        const Square sq = pop_lsb(queens);
        const Bitboard attacks = Bitboards::queen_attacks(sq, occupied);
        result.queen += popcount(attacks & ~ownPieces & ~enemyKing);
        result.safeQueen += popcount(attacks & ~ownPieces & ~enemyKing & ~enemyPawnAttacks);
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
        score += SAFE_MOBILITY_SCALE * (safe - mobility / 2);
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
