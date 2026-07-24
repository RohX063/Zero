#include "search.h"
#include "evaluation.h"
#include "movegen.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include "moveordering.h"
#include "tt.h"
#include "quiescence.h"
#include "zobrist.h"
#include "killer.h"
#include "history.h"

long long nodesSearched = 0;

int negamax(
    Board& board,
    int depth,
    int alpha,
    int beta,
    int ply
)
{
    Board original = board;
    bool sideAtEntry = board.isWhiteToMove();
    int alphaOriginal = alpha;

    //--------------------------------------------------
    // Transposition Table
    //--------------------------------------------------

    /*uint64_t hash = computeHash(board);

    TTEntry entry;

    if(probeTT(hash, entry))
    {
        if(entry.depth >= depth)
        {
            if(entry.flag == EXACT)
                return entry.score;

            if(entry.flag == LOWERBOUND)
                alpha = std::max(alpha, entry.score);

            if(entry.flag == UPPERBOUND)
                beta = std::min(beta, entry.score);

            if(alpha >= beta)
                return entry.score;
        }
    }*/

    nodesSearched++;

    //--------------------------------------------------
    // Base Case
    //--------------------------------------------------

    if(depth == 0)
    {
        return board.isWhiteToMove()
            ? evaluatePosition(board)
            : -evaluatePosition(board);
    }
     
    //----------------------------------------
    // Null Move Search
    //----------------------------------------

/*if(depth >= 3)
{
    board.makeNullMove();

    int score =
        -negamax(
            board,
            depth - 3,
            -beta,
            -beta + 1,
            ply + 1
        );

    board.undoNullMove();
}*/

    //--------------------------------------------------
    // Generate Moves
    //--------------------------------------------------

    std::vector<Move> moves =
        generateLegalMoves(
            board,
            board.isWhiteToMove()
        );

    for(Move &move : moves)
    {
        move.score =
            scoreMove(
                board,
                move,
                ply
            );
    }

    std::sort(
        moves.begin(),
        moves.end(),
        [](const Move &a, const Move &b)
        {
            return a.score > b.score;
        }
    );

    //--------------------------------------------------
    // No Legal Moves
    //--------------------------------------------------

    if(moves.empty())
    {
        if(board.isKingInCheck(board.isWhiteToMove()))
            return -100000;

        return 0;
    }

    //--------------------------------------------------
    // Search
    //--------------------------------------------------

    int bestScore = -1000000;

    bool firstMove = true;

    for(Move &move : moves)
    {
        bool isCapture =
            board.getPiece(
                move.toRow,
                move.toCol
            ) != EMPTY;

board.makeMove(move);

int score =
    -negamax(
        board,
        depth - 1,
        -beta,
        -alpha,
        ply + 1
    );

board.undoMove(move);

        if(score > bestScore)
            bestScore = score;

        //--------------------------------------------------
        // Move Ordering
        //--------------------------------------------------

        if(score > alpha)
            alpha = score;

        //--------------------------------------------------
        // Beta Cutoff
        //--------------------------------------------------

        if(alpha >= beta)
        {
            if(!isCapture)
            {
                /*addKillerMove(move, ply);

                addHistoryMove(move, depth);*/
            }

            break;
        }
    }

    //--------------------------------------------------
    // Store TT
    //--------------------------------------------------

    /*TTFlag flag;

    if(bestScore <= alphaOriginal)
        flag = UPPERBOUND;
    else if(bestScore >= beta)
        flag = LOWERBOUND;
    else
        flag = EXACT;

    storeTT(
        hash,
        depth,
        bestScore,
        flag,
        Move()
    );*/


    return bestScore;
}

Move findBestMove(Board& board, int depth)
{

    Board original = board;

    nodesSearched = 0;

    std::vector<Move> moves =
        generateLegalMoves(board, board.isWhiteToMove());

    Move bestMove;
    int bestScore = -1000000;

    //--------------------------------------------------
    // Move Ordering
    //--------------------------------------------------

    for (Move &move : moves)
    {
        move.score = scoreMove(board, move, 0);
    }


    std::sort(
        moves.begin(),
        moves.end(),
        [](const Move &a, const Move &b)
        {
            return a.score > b.score;
        }
    );

    //--------------------------------------------------
    // Search
    //--------------------------------------------------

for (Move &move : moves)
{
    Board before = board;

    board.makeMove(move);

    int score =
        -negamax(
            board,
            depth - 1,
            -1000000,
            1000000,
            0
        );

    board.undoMove(move);

    if (!board.equals(before))
    {
        std::cout << "\nROOT BOARD CORRUPTED after move ";

        std::cout
            << char('a' + move.fromCol)
            << char('8' - move.fromRow)
            << char('a' + move.toCol)
            << char('8' - move.toRow)
            << std::endl;
    }

    if (score > bestScore)
    {
        bestScore = score;
        bestMove = move;
        bestMove.score = score;
    }
    }

    return bestMove;
}

Move iterativeDeepening(
    Board& board,
    int maxDepth
)
{
    
    std::cout << "Searching...\n";
    clearHistory();

    Move bestMove;

    for(int depth=1; depth<=maxDepth; depth++)
    {
        Board beforeSearch = board;

        bestMove = findBestMove(board, depth);

std::cout
    << "Depth "
    << depth
    << " -> "
    << char('a' + bestMove.fromCol)
    << char('8' - bestMove.fromRow)
    << char('a' + bestMove.toCol)
    << char('8' - bestMove.toRow)
    << std::endl;
    }


    return bestMove;
}
