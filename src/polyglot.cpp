#include "polyglot.h"
#include "board.h"
#include "polyglot_keys.h"

uint64_t computePolyglotKey(const Board& board)
{
uint64_t hash = 0;

hash ^= POLYGLOT_RANDOM[0];

return hash;

}