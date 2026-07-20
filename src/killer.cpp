#include "killer.h"

Move killerMoves[MAX_PLY][2];

void addKillerMove(const Move& move, int ply)
{
    if(ply >= MAX_PLY)
        return;

    if(move.fromRow == killerMoves[ply][0].fromRow &&
       move.fromCol == killerMoves[ply][0].fromCol &&
       move.toRow   == killerMoves[ply][0].toRow &&
       move.toCol   == killerMoves[ply][0].toCol)
    {
        return;
    }

    killerMoves[ply][1] = killerMoves[ply][0];
    killerMoves[ply][0] = move;
}

bool isKillerMove(const Move& move, int ply)
{
    if(ply >= MAX_PLY)
        return false;

    for(int i=0;i<2;i++)
    {
        if(move.fromRow == killerMoves[ply][i].fromRow &&
           move.fromCol == killerMoves[ply][i].fromCol &&
           move.toRow   == killerMoves[ply][i].toRow &&
           move.toCol   == killerMoves[ply][i].toCol)
        {
            return true;
        }
    }

    return false;
}

void clearKillers()
{
    for(int ply = 0; ply < MAX_PLY; ++ply)
    {
        killerMoves[ply][0] = Move{};
        killerMoves[ply][1] = Move{};
    }
}