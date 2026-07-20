#ifndef KILLER_H
#define KILLER_H

#include "move.h"

const int MAX_PLY = 64;

extern Move killerMoves[MAX_PLY][2];

void addKillerMove(const Move& move, int ply);

bool isKillerMove(const Move& move, int ply);

void clearKillers();

#endif