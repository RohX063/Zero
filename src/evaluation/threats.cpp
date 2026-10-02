#include "threats.h"

#include "../piece.h"
#include "../position.h"
#include "material.h"

#include <algorithm>
#include <array>

namespace Zero::Evaluation {

namespace {

constexpr int ATTACKED_PIECE_BONUS = 2;
constexpr int HANGING_SCALE = 7;
constexpr int LOOSE_SCALE = 1;
constexpr int MULTI_ATTACK_BONUS = 7;
constexpr int PAWN_THREAT_BONUS = 5;
constexpr int FORK_BONUS = 18;
constexpr int HIGH_VALUE_FORK_BONUS = 30;
constexpr int PIN_BONUS = 12;
constexpr int XRAY_BONUS = 8;
constexpr int SKEWER_BONUS = 24;
constexpr int PROMOTION_BONUS = 35;
constexpr int CONVERGENCE_BONUS = 2;

constexpr std::array<int, 8> df = {0, 0, 1, -1, 1, -1, 1, -1};
constexpr std::array<int, 8> dr = {1, -1, 0, 0, 1, 1, -1, -1};

inline bool inBoard(int file, int rank)
{
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

inline bool isSliderForDirection(PieceType type, bool orthogonal)
{
    return orthogonal
        ? (type == ROOK || type == QUEEN)
        : (type == BISHOP || type == QUEEN);
}

inline bool isHighValue(Piece piece)
{
    const PieceType type = type_of(piece);
    return type == ROOK || type == QUEEN;
}

int countAbsolutePins(const Position& position, Color attacker)
{
    const Color victim = ~attacker;
    const Square king = position.king_square(victim);
    if (king == SQ_NONE)
        return 0;

    int pins = 0;
    const int kingFile = file_of(king);
    const int kingRank = rank_of(king);

    for (int dir = 0; dir < 8; ++dir) {
        int file = kingFile + df[dir];
        int rank = kingRank + dr[dir];
        if (!inBoard(file, rank))
            continue;

        // First blocker from the king must be a victim piece.
        while (inBoard(file, rank) &&
               position.piece_on(make_square(file, rank)) == EMPTY) {
            file += df[dir];
            rank += dr[dir];
        }
        if (!inBoard(file, rank))
            continue;

        const Piece pinned = position.piece_on(make_square(file, rank));
        if (pinned == EMPTY || color_of(pinned) != victim || type_of(pinned) == KING)
            continue;

        // Second blocker must be an attacker slider compatible with the ray.
        file += df[dir];
        rank += dr[dir];
        while (inBoard(file, rank) &&
               position.piece_on(make_square(file, rank)) == EMPTY) {
            file += df[dir];
            rank += dr[dir];
        }
        if (!inBoard(file, rank))
            continue;

        const Piece pinner = position.piece_on(make_square(file, rank));
        if (pinner == EMPTY || color_of(pinner) != attacker)
            continue;

        if (isSliderForDirection(type_of(pinner), dir < 4))
            ++pins;
    }

    return pins;
}

int countXrayTargets(const Position& position, Color attacker)
{
    const Color enemy = ~attacker;
    int count = 0;

    const PieceType sliderTypes[] = {BISHOP, ROOK, QUEEN};
    for (PieceType sliderType : sliderTypes) {
        Bitboard sliders = position.pieces(attacker, sliderType);
        while (sliders) {
            const Square source = pop_lsb(sliders);
            const int sourceFile = file_of(source);
            const int sourceRank = rank_of(source);

            for (int dir = 0; dir < 8; ++dir) {
                const bool orthogonal = dir < 4;
                if (!isSliderForDirection(sliderType, orthogonal))
                    continue;

                int file = sourceFile + df[dir];
                int rank = sourceRank + dr[dir];
                if (!inBoard(file, rank))
                    continue;

                while (inBoard(file, rank) &&
                       position.piece_on(make_square(file, rank)) == EMPTY) {
                    file += df[dir];
                    rank += dr[dir];
                }
                if (!inBoard(file, rank))
                    continue;

                const Piece blocker = position.piece_on(make_square(file, rank));
                if (blocker == EMPTY || color_of(blocker) != enemy || type_of(blocker) == KING)
                    continue;

                file += df[dir];
                rank += dr[dir];
                while (inBoard(file, rank) &&
                       position.piece_on(make_square(file, rank)) == EMPTY) {
                    file += df[dir];
                    rank += dr[dir];
                }
                if (!inBoard(file, rank))
                    continue;

                const Piece target = position.piece_on(make_square(file, rank));
                if (target == EMPTY || color_of(target) != enemy || type_of(target) == KING)
                    continue;

                if (pieceValue(target) >= pieceValue(blocker))
                    ++count;
            }
        }
    }

    return count;
}

int countSkewers(const Position& position, Color attacker)
{
    const Color enemy = ~attacker;
    const Square king = position.king_square(enemy);
    if (king == SQ_NONE)
        return 0;

    int skewers = 0;
    const int kingFile = file_of(king);
    const int kingRank = rank_of(king);

    // A skewer exists statically when the king is the front target on a clear
    // slider line and a second enemy unit lies behind the king on that line.
    for (int dir = 0; dir < 8; ++dir) {
        const bool orthogonal = dir < 4;

        int file = kingFile + df[dir];
        int rank = kingRank + dr[dir];
        if (!inBoard(file, rank))
            continue;

        while (inBoard(file, rank) &&
               position.piece_on(make_square(file, rank)) == EMPTY) {
            file += df[dir];
            rank += dr[dir];
        }
        if (!inBoard(file, rank))
            continue;

        const Piece target = position.piece_on(make_square(file, rank));
        if (target == EMPTY || color_of(target) != enemy || type_of(target) == KING)
            continue;

        int af = kingFile - df[dir];
        int ar = kingRank - dr[dir];
        if (!inBoard(af, ar))
            continue;

        while (inBoard(af, ar) &&
               position.piece_on(make_square(af, ar)) == EMPTY) {
            af -= df[dir];
            ar -= dr[dir];
        }
        if (!inBoard(af, ar))
            continue;

        const Piece slider = position.piece_on(make_square(af, ar));
        if (slider != EMPTY && color_of(slider) == attacker &&
            isSliderForDirection(type_of(slider), orthogonal))
            ++skewers;
    }

    return skewers;
}

int countPromotionThreats(const Position& position, Color color)
{
    const Color enemy = ~color;
    const Bitboard enemyPieces = position.pieces(enemy);
    Bitboard pawns = position.pieces(color, PAWN);
    int threats = 0;

    while (pawns) {
        const Square sq = pop_lsb(pawns);
        const int file = file_of(sq);
        const int rank = rank_of(sq);
        const int promotionRank = color == WHITE ? 7 : 0;
        if ((color == WHITE && rank != 6) || (color == BLACK && rank != 1))
            continue;

        const int nextRank = color == WHITE ? rank + 1 : rank - 1;
        if (nextRank != promotionRank)
            continue;

        const Square forward = make_square(file, nextRank);
        if (position.piece_on(forward) == EMPTY)
            ++threats;

        if (file > 0 && (enemyPieces & square_bb(make_square(file - 1, nextRank))))
            ++threats;
        if (file < 7 && (enemyPieces & square_bb(make_square(file + 1, nextRank))))
            ++threats;
    }

    return threats;
}

int countForks(const Position& position, Color color, int& highValueForks)
{
    const Color enemy = ~color;
    const Bitboard enemyTargets = position.pieces(enemy)
                                & ~position.pieces(enemy, PAWN)
                                & ~position.pieces(enemy, KING);
    Bitboard knights = position.pieces(color, KNIGHT);
    int forks = 0;
    highValueForks = 0;

    while (knights) {
        const Square sq = pop_lsb(knights);
        const Bitboard targets = Bitboards::KnightAttacks[sq] & enemyTargets;
        if (popcount(targets) < 2)
            continue;

        ++forks;
        Bitboard scan = targets;
        while (scan) {
            const Square targetSq = pop_lsb(scan);
            if (isHighValue(position.piece_on(targetSq))) {
                ++highValueForks;
                break;
            }
        }
    }

    return forks;
}

} // namespace

ThreatFeatures analyzeThreats(const Position& position, Color color)
{
    ThreatFeatures result;
    const Color enemy = ~color;

    const PieceType victimTypes[] = {KNIGHT, BISHOP, ROOK, QUEEN};
    int hangingValue = 0;
    int looseValue = 0;

    for (PieceType victimType : victimTypes) {
        Bitboard victims = position.pieces(enemy, victimType);
        while (victims) {
            const Square sq = pop_lsb(victims);
            const Bitboard attackers = position.attackers_to(sq, color);
            if (!attackers)
                continue;

            ++result.attackedPieces;
            const Piece victim = position.piece_on(sq);
            const Bitboard defenders = position.attackers_to(sq, enemy);
            const int victimValue = pieceValue(victim);

            if (!defenders) {
                ++result.loosePieces;
                ++result.hangingPieces;
                looseValue += victimValue;
                hangingValue += victimValue;
            }

            if (popcount(attackers) >= 2)
                ++result.multiAttackedPieces;
        }
    }

    Bitboard pawns = position.pieces(color, PAWN);
    while (pawns) {
        const Square sq = pop_lsb(pawns);
        Bitboard targets = Bitboards::PawnAttacks[color][sq]
                         & position.pieces(enemy)
                         & ~position.pieces(enemy, PAWN);
        while (targets) {
            const Square targetSq = pop_lsb(targets);
            if (type_of(position.piece_on(targetSq)) != KING)
                ++result.pawnThreats;
        }
    }

    result.forks = countForks(position, color, result.highValueForks);
    result.pins = countAbsolutePins(position, color);
    result.xrayTargets = countXrayTargets(position, color);
    result.skewers = countSkewers(position, color);
    result.promotionThreats = countPromotionThreats(position, color);

    const int convergence = std::max(0,
        result.attackedPieces
        + result.multiAttackedPieces
        + result.forks
        + result.pins
        - 2);

    result.score = ATTACKED_PIECE_BONUS * result.attackedPieces
                 + (HANGING_SCALE * hangingValue) / 100
                 + (LOOSE_SCALE * looseValue) / 100
                 + MULTI_ATTACK_BONUS * result.multiAttackedPieces
                 + PAWN_THREAT_BONUS * result.pawnThreats
                 + FORK_BONUS * result.forks
                 + HIGH_VALUE_FORK_BONUS * result.highValueForks
                 + PIN_BONUS * result.pins
                 + XRAY_BONUS * result.xrayTargets
                 + SKEWER_BONUS * result.skewers
                 + PROMOTION_BONUS * result.promotionThreats
                 + CONVERGENCE_BONUS * convergence * convergence;

    return result;
}

int evaluateThreats(const Position& position)
{
    return analyzeThreats(position, WHITE).score
         - analyzeThreats(position, BLACK).score;
}

} // namespace Zero::Evaluation
