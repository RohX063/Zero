#include "king.h"

#include "../piece.h"
#include "../position.h"

#include <algorithm>
#include <array>
#include <cstdlib>

namespace Zero::Evaluation {

namespace {

constexpr int MAX_PHASE = 20;
constexpr int FULL_PHASE = 256;
constexpr int MIN_SAFETY_SCALE = 64; // Keep a residual king-safety signal in late endgames.

// These weights are intentionally compact and live in one translation unit so
// future tuning remains localized. They are not intended to mimic Stockfish's
// trained NNUE parameters; this is ZERO's own classical king-safety model.
constexpr int SHIELD_FIRST_RANK = 9;
constexpr int SHIELD_SECOND_RANK = 5;
constexpr int MISSING_SHIELD = 11;
constexpr int PAWN_STORM = 4;
constexpr int RING_SQUARE = 4;
constexpr int RING_ATTACKER = 5;
constexpr int PAWN_RING = 4;
constexpr int HEAVY_ATTACKER = 8;
constexpr int DIRECT_ATTACKER = 16;
constexpr int XRAY_WEIGHT = 6;
constexpr int SAFE_ESCAPE = 7;
constexpr int UNSAFE_ESCAPE = 2;

constexpr int rookPhase = 2;
constexpr int queenPhase = 4;

constexpr std::array<int, 8> df = {0, 0, 1, -1, 1, -1, 1, -1};
constexpr std::array<int, 8> dr = {1, -1, 0, 0, 1, 1, -1, -1};

inline bool inBoard(int file, int rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

int gamePhase256(const Position& position)
{
    using namespace Zero;

    int phase = 0;
    phase += popcount(position.pieces(WHITE, KNIGHT));
    phase += popcount(position.pieces(BLACK, KNIGHT));
    phase += popcount(position.pieces(WHITE, BISHOP));
    phase += popcount(position.pieces(BLACK, BISHOP));
    phase += rookPhase * popcount(position.pieces(WHITE, ROOK));
    phase += rookPhase * popcount(position.pieces(BLACK, ROOK));
    phase += queenPhase * popcount(position.pieces(WHITE, QUEEN));
    phase += queenPhase * popcount(position.pieces(BLACK, QUEEN));

    phase = std::min(MAX_PHASE, phase);
    return phase * FULL_PHASE / MAX_PHASE;
}

int scaleKingSafety(int value, int phase256)
{
    const int scale = MIN_SAFETY_SCALE
                    + (FULL_PHASE - MIN_SAFETY_SCALE) * phase256 / FULL_PHASE;
    return value * scale / FULL_PHASE;
}

Bitboard attackersTo(const Position& position,
                     Square target,
                     Color by,
                     Bitboard occupied)
{
    using namespace Zero;

    Bitboard attackers = Bitboards::PawnAttacks[~by][target]
                       & position.pieces(by, PAWN);

    attackers |= Bitboards::KnightAttacks[target]
              & position.pieces(by, KNIGHT);

    attackers |= Bitboards::KingAttacks[target]
              & position.pieces(by, KING);

    attackers |= Bitboards::bishop_attacks(target, occupied)
              & position.pieces(by, BISHOP);
    attackers |= Bitboards::bishop_attacks(target, occupied)
              & position.pieces(by, QUEEN);

    attackers |= Bitboards::rook_attacks(target, occupied)
              & position.pieces(by, ROOK);
    attackers |= Bitboards::rook_attacks(target, occupied)
              & position.pieces(by, QUEEN);

    return attackers;
}

Bitboard pieceAttacks(Square square,
                      PieceType type,
                      Bitboard occupied,
                      Color color)
{
    using namespace Zero;

    switch (type) {
        case PAWN:   return Bitboards::PawnAttacks[color][square];
        case KNIGHT: return Bitboards::KnightAttacks[square];
        case BISHOP: return Bitboards::bishop_attacks(square, occupied);
        case ROOK:   return Bitboards::rook_attacks(square, occupied);
        case QUEEN:  return Bitboards::queen_attacks(square, occupied);
        case KING:   return Bitboards::KingAttacks[square];
        default:     return 0;
    }
}

struct RingPressure {
    Bitboard attackedSquares = 0;
    int attackers = 0;
    int pawnAttacks = 0;
    int heavyAttackers = 0;
};

RingPressure calculateRingPressure(const Position& position,
                                   Square kingSquare,
                                   Color enemy,
                                   Bitboard occupied)
{
    using namespace Zero;

    RingPressure result;
    const Bitboard ring = Bitboards::KingAttacks[kingSquare];

    const PieceType types[] = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};
    for (PieceType type : types) {
        Bitboard pieces = position.pieces(enemy, type);
        while (pieces) {
            const Square sq = pop_lsb(pieces);
            const Bitboard attacks = pieceAttacks(sq, type, occupied, enemy);
            const Bitboard ringHits = attacks & ring;
            if (!ringHits)
                continue;

            ++result.attackers;
            result.attackedSquares |= ringHits;

            if (type == PAWN)
                result.pawnAttacks += popcount(ringHits);
            else if (type == ROOK || type == QUEEN)
                ++result.heavyAttackers;
        }
    }

    return result;
}

int countPawnStorm(const Position& position,
                   Square kingSquare,
                   Color enemy)
{
    using namespace Zero;

    const Bitboard enemyPawns = position.pieces(enemy, PAWN);
    Bitboard pawns = enemyPawns;
    int storm = 0;

    const int kingFile = file_of(kingSquare);
    const int kingRank = rank_of(kingSquare);

    while (pawns) {
        const Square sq = pop_lsb(pawns);
        const int fileDistance = std::abs(file_of(sq) - kingFile);
        const int rankDistance = std::abs(rank_of(sq) - kingRank);

        if (fileDistance <= 1 && rankDistance <= 2)
            ++storm;
    }

    return storm;
}

void analyzeShield(const Position& position,
                   Square kingSquare,
                   Color color,
                   KingSafetyFeatures& result)
{
    using namespace Zero;

    const Bitboard friendlyPawns = position.pieces(color, PAWN);
    const int kingFile = file_of(kingSquare);
    const int kingRank = rank_of(kingSquare);
    const int direction = color == WHITE ? 1 : -1;

    for (int offset = -1; offset <= 1; ++offset) {
        const int file = kingFile + offset;
        if (file < 0 || file >= 8) {
            ++result.missingShield;
            continue;
        }

        const int firstRank = kingRank + direction;
        const int secondRank = kingRank + 2 * direction;

        bool firstFound = false;
        if (firstRank >= 0 && firstRank < 8) {
            firstFound = (friendlyPawns & square_bb(make_square(file, firstRank))) != 0;
            if (firstFound) {
                ++result.shieldPawns;
                result.shieldStrength += SHIELD_FIRST_RANK;
            }
        }

        if (!firstFound && secondRank >= 0 && secondRank < 8) {
            const bool secondFound = (friendlyPawns & square_bb(make_square(file, secondRank))) != 0;
            if (secondFound) {
                ++result.shieldPawns;
                result.shieldStrength += SHIELD_SECOND_RANK;
            }
        }

        if (!firstFound
            && (secondRank < 0 || secondRank >= 8
                || (friendlyPawns & square_bb(make_square(file, secondRank))) == 0))
            ++result.missingShield;
    }
}

int countXrayPressure(const Position& position,
                      Square kingSquare,
                      Color color)
{
    using namespace Zero;

    const Color enemy = ~color;
    int score = 0;

    const int kingFile = file_of(kingSquare);
    const int kingRank = rank_of(kingSquare);

    for (int dir = 0; dir < 8; ++dir) {
        int file = kingFile + df[dir];
        int rank = kingRank + dr[dir];
        if (!inBoard(file, rank))
            continue;

        // Locate the first occupied square on the ray. If it belongs to the
        // defending side, continue to the next occupied square and look for
        // a compatible enemy slider. This models an actual x-ray through a
        // single shielding unit rather than only the adjacent-square case.
        while (inBoard(file, rank) && position.piece_on(make_square(file, rank)) == EMPTY) {
            file += df[dir];
            rank += dr[dir];
        }
        if (!inBoard(file, rank))
            continue;

        const Piece firstBlocker = position.piece_on(make_square(file, rank));
        if (firstBlocker == EMPTY || color_of(firstBlocker) != color)
            continue;

        file += df[dir];
        rank += dr[dir];
        while (inBoard(file, rank) && position.piece_on(make_square(file, rank)) == EMPTY) {
            file += df[dir];
            rank += dr[dir];
        }
        if (!inBoard(file, rank))
            continue;

        const Piece behind = position.piece_on(make_square(file, rank));
        if (behind == EMPTY || color_of(behind) != enemy)
            continue;

        const PieceType type = type_of(behind);
        const bool orthogonal = dir < 4;
        const bool matchingSlider = orthogonal
            ? (type == ROOK || type == QUEEN)
            : (type == BISHOP || type == QUEEN);

        if (matchingSlider)
            score += (type == QUEEN ? 2 : 1);
    }

    return score;
}

int kingProximityPressure(const Position& position,
                          Square kingSquare,
                          Color enemy)
{
    using namespace Zero;

    const int kingFile = file_of(kingSquare);
    const int kingRank = rank_of(kingSquare);
    int pressure = 0;

    const PieceType types[] = {KNIGHT, BISHOP, ROOK, QUEEN};
    for (PieceType type : types) {
        Bitboard pieces = position.pieces(enemy, type);
        while (pieces) {
            const Square sq = pop_lsb(pieces);
            const int distance = std::max(std::abs(file_of(sq) - kingFile),
                                          std::abs(rank_of(sq) - kingRank));
            if (distance <= 2)
                pressure += (type == QUEEN ? 2 : 1);
        }
    }

    return pressure;
}

} // namespace

KingSafetyFeatures analyzeKingSafety(const Position& position, Color color)
{
    using namespace Zero;

    KingSafetyFeatures result;
    result.kingSquare = position.king_square(color);
    result.phase256 = gamePhase256(position);

    if (result.kingSquare == SQ_NONE)
        return result;

    const Color enemy = ~color;
    const Bitboard occupied = position.pieces();
    const Bitboard kingBit = square_bb(result.kingSquare);
    const Bitboard ownPieces = position.pieces(color);
    const Bitboard candidateEscapes = Bitboards::KingAttacks[result.kingSquare] & ~ownPieces;

    analyzeShield(position, result.kingSquare, color, result);

    const RingPressure pressure = calculateRingPressure(
        position, result.kingSquare, enemy, occupied);

    result.ringAttackedSquares = popcount(pressure.attackedSquares);
    result.ringAttackers = pressure.attackers;
    result.pawnRingAttacks = pressure.pawnAttacks;
    result.heavyRingAttackers = pressure.heavyAttackers;
    result.directAttackers = popcount(
        attackersTo(position, result.kingSquare, enemy, occupied));
    result.xrayPressure = countXrayPressure(position, result.kingSquare, color);
    result.pawnStorm = countPawnStorm(position, result.kingSquare, enemy);

    // Escape-square analysis removes the king and a captured enemy piece from
    // occupancy before testing attacks. This is closer to a real king-move
    // legality probe than simply asking Position::isSquareAttacked() on the
    // current board.
    const Bitboard occupiedWithoutKing = occupied & ~kingBit;
    Bitboard escapes = candidateEscapes;
    while (escapes) {
        const Square target = pop_lsb(escapes);
        const Bitboard targetBit = square_bb(target);
        const Bitboard occupancyAfterKingMove =
            occupiedWithoutKing & ~targetBit;

        const Bitboard attacks = attackersTo(
            position, target, enemy, occupancyAfterKingMove);

        if (attacks == 0)
            ++result.safeEscapeSquares;
        else
            ++result.unsafeEscapeSquares;

    }

    // The raw danger model deliberately mixes several independent signals:
    // shelter, king-ring saturation, direct pressure, x-rays and escape
    // geometry. A small nonlinear reinforcement makes converging attacks more
    // dangerous than isolated pressure without turning the evaluator into a
    // mate detector.
    const int ringDanger =
          RING_SQUARE    * result.ringAttackedSquares
        + RING_ATTACKER  * result.ringAttackers
        + PAWN_RING      * result.pawnRingAttacks
        + HEAVY_ATTACKER * result.heavyRingAttackers
        + DIRECT_ATTACKER * result.directAttackers
        + XRAY_WEIGHT    * result.xrayPressure
        + PAWN_STORM     * result.pawnStorm
        + UNSAFE_ESCAPE  * result.unsafeEscapeSquares;

    const int convergingAttack = std::max(0,
        result.ringAttackers + result.directAttackers - 2);
    const int nonlinearDanger = convergingAttack * convergingAttack;

    const int rawSafety = result.shieldStrength
                        - MISSING_SHIELD * result.missingShield
                        - ringDanger
                        - nonlinearDanger
                        + SAFE_ESCAPE * result.safeEscapeSquares;

    // Enemy pieces physically approaching the king matter, but only as a small
    // supplement because attack maps already capture the stronger tactical fact.
    const int proximity = kingProximityPressure(position, result.kingSquare, enemy);
    result.score = scaleKingSafety(rawSafety - 2 * proximity, result.phase256);

    return result;
}

int evaluateKingSafety(const Position& position)
{
    return analyzeKingSafety(position, WHITE).score
         - analyzeKingSafety(position, BLACK).score;
}

} // namespace Zero::Evaluation
