#include "pawns.h"

#include "../position.h"

#include <algorithm>

namespace Zero::Evaluation {

namespace {

constexpr int DOUBLED_PAWN_PENALTY = 12;
constexpr int ISOLATED_PAWN_PENALTY = 10;
constexpr int BACKWARD_PAWN_PENALTY = 8;
constexpr int CONNECTED_PAWN_BONUS = 3;
constexpr int PAWN_ISLAND_PENALTY = 2;
constexpr int PASSED_PAWN_BASE_BONUS = 12;
constexpr int PASSED_PAWN_ADVANCE_BONUS = 4;

Bitboard adjacentFiles(int file)
{
    Bitboard files = 0;
    if (file > 0)
        files |= Bitboards::FileBB[file - 1];
    if (file < 7)
        files |= Bitboards::FileBB[file + 1];
    return files;
}

Bitboard sameAndAdjacentFiles(int file)
{
    return Bitboards::FileBB[file] | adjacentFiles(file);
}

Bitboard rankBandAround(int rank)
{
    Bitboard ranks = Bitboards::RankBB[rank];
    if (rank > 0)
        ranks |= Bitboards::RankBB[rank - 1];
    if (rank < 7)
        ranks |= Bitboards::RankBB[rank + 1];
    return ranks;
}

Bitboard ranksBehind(int rank, Color color)
{
    Bitboard ranks = 0;

    if (color == WHITE) {
        for (int r = 0; r <= rank; ++r)
            ranks |= Bitboards::RankBB[r];
    } else {
        for (int r = rank; r < 8; ++r)
            ranks |= Bitboards::RankBB[r];
    }

    return ranks;
}

bool isPassedPawn(Square sq, Bitboard enemyPawns, Color color)
{
    const int file = file_of(sq);
    const int rank = rank_of(sq);
    const Bitboard enemyFiles = sameAndAdjacentFiles(file);

    Bitboard enemyAhead = 0;
    if (color == WHITE) {
        for (int r = rank + 1; r < 8; ++r)
            enemyAhead |= Bitboards::RankBB[r];
    } else {
        for (int r = rank - 1; r >= 0; --r)
            enemyAhead |= Bitboards::RankBB[r];
    }

    return (enemyPawns & enemyFiles & enemyAhead) == 0;
}

bool isBackwardPawn(Square sq, Bitboard friendlyPawns, Bitboard enemyPawns, Color color)
{
    const int file = file_of(sq);
    const int rank = rank_of(sq);
    const Bitboard neighbors = adjacentFiles(file);
    const Bitboard supportingRanks = ranksBehind(rank, color);

    if (friendlyPawns & neighbors & supportingRanks)
        return false;

    const int advanceRank = color == WHITE ? rank + 1 : rank - 1;
    if (advanceRank < 0 || advanceRank > 7)
        return false;

    const Bitboard enemyAdjacent =
        enemyPawns & adjacentFiles(file) & Bitboards::RankBB[advanceRank];

    if (enemyAdjacent == 0)
        return false;

    return !isPassedPawn(sq, enemyPawns, color);
}

int pawnRelativeRank(Square sq, Color color)
{
    const int rank = rank_of(sq);
    return color == WHITE ? rank + 1 : 8 - rank;
}

int evaluateFeatures(const PawnFeatures& f)
{
    return -DOUBLED_PAWN_PENALTY * f.doubled
         -ISOLATED_PAWN_PENALTY * f.isolated
         -BACKWARD_PAWN_PENALTY * f.backward
         +CONNECTED_PAWN_BONUS * f.connected
         -PAWN_ISLAND_PENALTY * std::max(0, f.islands - 1);
}

int evaluateSide(Bitboard friendlyPawns, Bitboard enemyPawns, Color color)
{
    const PawnFeatures f = analyzePawnStructure(friendlyPawns, enemyPawns, color);
    return evaluateFeatures(f);
}

} // namespace

PawnFeatures analyzePawnStructure(Bitboard friendlyPawns, Bitboard enemyPawns, Color color)
{
    PawnFeatures f;
    f.pawns = popcount(friendlyPawns);

    int fileCount[8]{};
    Bitboard pawns = friendlyPawns;

    while (pawns) {
        const Square sq = pop_lsb(pawns);
        ++fileCount[file_of(sq)];

        const int file = file_of(sq);
        const int rank = rank_of(sq);
        const Bitboard neighbors = adjacentFiles(file);

        if ((friendlyPawns & neighbors) == 0)
            ++f.isolated;

        const Bitboard connectedArea = neighbors & rankBandAround(rank);
        if (friendlyPawns & connectedArea)
            ++f.connected;

        if (isBackwardPawn(sq, friendlyPawns, enemyPawns, color))
            ++f.backward;

        if (isPassedPawn(sq, enemyPawns, color))
            ++f.passed;
    }

    for (int file = 0; file < 8; ++file)
        if (fileCount[file] > 1)
            f.doubled += fileCount[file] - 1;

    bool previousFileHadPawn = false;
    for (int file = 0; file < 8; ++file) {
        const bool hasPawn = fileCount[file] != 0;
        if (hasPawn && !previousFileHadPawn)
            ++f.islands;
        previousFileHadPawn = hasPawn;
    }

    return f;
}

int evaluatePawns(const Position& position)
{
    using namespace Zero;

    const Bitboard whitePawns = position.pieces(WHITE, PAWN);
    const Bitboard blackPawns = position.pieces(BLACK, PAWN);

    auto sideScore = [&](Bitboard friendly, Bitboard enemy, Color color) {
        const PawnFeatures f = analyzePawnStructure(friendly, enemy, color);
        int score = evaluateFeatures(f);

        Bitboard passed = friendly;
        while (passed) {
            const Square sq = pop_lsb(passed);
            if (!isPassedPawn(sq, enemy, color))
                continue;

            const int rr = pawnRelativeRank(sq, color);
            score += PASSED_PAWN_BASE_BONUS;

            const int advanceRank = color == WHITE
                ? rank_of(sq) + 1
                : rank_of(sq) - 1;

            const bool blocked = advanceRank < 0 || advanceRank > 7
                || position.piece_on(make_square(file_of(sq), advanceRank)) != EMPTY;

            // Advancement is valuable only when the pawn actually has a
            // forward route. A blocked passed pawn remains structurally useful,
            // but it should not receive dynamic advancement credit.
            if (!blocked)
                score += PASSED_PAWN_ADVANCE_BONUS * std::max(0, rr - 3);
        }

        return score;
    };

    return sideScore(whitePawns, blackPawns, WHITE)
         - sideScore(blackPawns, whitePawns, BLACK);
}
} // namespace Zero::Evaluation
