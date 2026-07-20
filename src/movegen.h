#ifndef MOVEGEN_H
#define MOVEGEN_H

#include <vector>

#include "board.h"
#include "move.h"

std::vector<Move> generateAllMoves(const Board& board, bool whiteToMove);

std::vector<Move> generateLegalMoves(
    Board& board,
    bool whiteToMove
);

bool isCheckmate(
    Board& board,
    bool whiteToMove
);

std::vector<Move> generateCaptureMoves(
    Board& board,
    bool whiteToMove
);

#endif