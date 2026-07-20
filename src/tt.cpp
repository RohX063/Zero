#include "tt.h"
#include <unordered_map>
#include <iostream>

std::unordered_map<uint64_t, TTEntry> TT;

bool probeTT(
    uint64_t hash,
    TTEntry& entry
)
{
    auto it = TT.find(hash);

    if(it == TT.end())
    {
        return false;
    }

    entry = it->second;

    return true;
}

void storeTT(
    uint64_t hash,
    int depth,
    int score,
    TTFlag flag,
    const Move& bestMove
)
{
    TTEntry entry;

    entry.hash = hash;
    entry.depth = depth;
    entry.score = score;
    entry.flag = flag;
    entry.bestMove = bestMove;

    TT[hash] = entry;
}