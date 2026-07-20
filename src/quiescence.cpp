#include "quiescence.h"
#include <vector>
#include "evaluation.h"
#include <algorithm>
#include "moveordering.h"
#include <iostream>
#include "movegen.h"

int quiescence(
    Board& board,
    int alpha,
    int beta
)
{
    int standPat =
        board.isWhiteToMove()
        ? evaluatePosition(board)
        : -evaluatePosition(board);

    if(standPat >= beta)
    {
        return beta;
    }

    if(standPat > alpha)
    {
        alpha = standPat;
    }

    std::vector<Move> captures =
    generateCaptureMoves(
        board,
        board.isWhiteToMove()
    );

    std::sort(
    captures.begin(),
    captures.end(),

    [&](const Move& a, const Move& b)
    {
        return scoreMove(board, a, 0)
             > scoreMove(board, b, 0);
    }
);

    for(Move& move : captures)
{

    board.makeMove(move);
    
    int score =
        -quiescence(
            board,
            -beta,
            -alpha
        );

    board.undoMove(move);

    if(score >= beta)
    {
        return beta;
    }

    if(score > alpha)
    {
        alpha = score;
    }
}

    return alpha;
}