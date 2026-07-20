#include "zobrist.h"
#include "board.h"
#include <random>
#include "piece.h"

uint64_t zobristTable[12][8][8];
uint64_t sideToMoveKey;

void initializeZobrist()
{
  std::mt19937_64 rng(2026);
  std::uniform_int_distribution<uint64_t> dist;
  
  for(int piece = 0; piece < 12; piece++)
    {
    for(int row = 0; row < 8; row++)
    {
        for(int col = 0; col < 8; col++)
        {
            zobristTable[piece][row][col] = dist(rng);
        }
    }
    }

    sideToMoveKey = dist(rng);
} 

uint64_t computeHash(const Board& board)
{
    uint64_t hash = 0;

    if(board.isWhiteToMove())
    {
       hash ^= sideToMoveKey;
    }

    for(int row = 0; row < 8; row++)
    {
        for(int col = 0; col < 8; col++)
        {
            Piece piece = board.getPiece(row, col);
            if(piece != EMPTY)
            {
                hash ^= zobristTable[piece - 1][row][col];
            }
        }
    }

    return hash;
}