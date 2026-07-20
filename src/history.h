#ifndef HISTORY_H
#define HISTORY_H

#include "move.h"

extern int historyTable[64][64];

void clearHistory();

void addHistoryMove(
    const Move& move,
    int depth
);

int getHistoryScore(
    const Move& move
);

#endif