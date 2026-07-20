#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include "move.h"

Move findBestMove(Board& board, int depth);
Move iterativeDeepening(Board& board, int maxDepth);

int negamax(
    Board& board,
    int depth,
    int alpha,
    int beta,
    int ply
);
extern long long nodesSearched;

#endif