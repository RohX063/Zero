#include "history.h"

int historyTable[64][64];

void clearHistory()
{
    for(int i = 0; i < 64; i++)
    {
        for(int j = 0; j < 64; j++)
        {
            historyTable[i][j] = 0;
        }
    }
}

void addHistoryMove(
    const Move& move,
    int depth
)
{
    int from =
        move.fromRow * 8
        + move.fromCol;

    int to =
        move.toRow * 8
        + move.toCol;

    historyTable[from][to] += depth * depth;

    if(historyTable[from][to] > 100000)
    {
        historyTable[from][to] = 100000;
    }
}

int getHistoryScore(
    const Move& move
)
{
    int from =
        move.fromRow * 8
        + move.fromCol;

    int to =
        move.toRow * 8
        + move.toCol;

    return historyTable[from][to];
}