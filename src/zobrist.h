#pragma once

#include <cstdint>

class Board;

extern uint64_t zobristTable[12][8][8];
extern uint64_t sideToMoveKey;

uint64_t computeHash(const Board& board);

void initializeZobrist();