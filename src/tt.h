#ifndef TT_H
#define TT_H

#include <cstdint>
#include <unordered_map>
#include "move.h"

enum TTFlag
{
    EXACT,
    LOWERBOUND,
    UPPERBOUND
};

struct TTEntry
{
    uint64_t hash;

    int depth;
    int score;

    TTFlag flag;

    Move bestMove;
};

extern std::unordered_map<uint64_t, TTEntry> TT;

bool probeTT(
    uint64_t hash,
    TTEntry& entry
);

void storeTT(
    uint64_t hash,
    int depth,
    int score,
    TTFlag flag,
    const Move& bestMove
);

#endif