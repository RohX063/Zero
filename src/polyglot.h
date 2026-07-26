#pragma once

#include <cstdint>

class Board;

uint64_t computePolyglotKey(const Board& board);

#pragma pack(push, 1)

struct PolyglotEntry
{
    uint64_t key;      // 8 bytes
    uint16_t move;     // 2 bytes
    uint16_t weight;   // 2 bytes
    uint32_t learn;    // 4 bytes
};

#pragma pack(pop)