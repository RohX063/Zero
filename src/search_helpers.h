#ifndef ZERO_SEARCH_HELPERS_H
#define ZERO_SEARCH_HELPERS_H

#include "move.h"
#include "piece.h"
#include "types.h"
#include "search_params.h"
#include "search_types.h"

#include <string>

class Position;

namespace Zero::Search {

struct TriedMove {
    Move move{};
    Piece attacker = EMPTY;
    Piece victim = EMPTY;
    bool quiet = false;
};

bool isQuietMove(const Move& move, Piece victim);
Piece capturedPiece(const Position& position, const Move& move);
int historyBonus(const SearchParams& params, Depth depth);
std::string uciMoveString(const Move& move);
std::string uciScoreString(Value score);
int evaluateLMRFormula(const LMRParameters& params, int depth, int moveCount);

} // namespace Zero::Search

#endif
