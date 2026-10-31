#ifndef ZERO_MOVEGEN_H
#define ZERO_MOVEGEN_H

#include <array>
#include <cstddef>
#include <vector>
#include "move.h"
#include "types.h"

class Position;

inline constexpr std::size_t MAX_MOVES = 256;
inline constexpr std::size_t MAX_TACTICAL_MOVES = 256;

template<std::size_t Capacity>
struct MoveList {
    static_assert(Capacity > 0, "MoveList capacity must be non-zero");

    std::array<Move, Capacity> moves{};
    std::size_t count = 0;

    void push(Move move) {
        if (count < Capacity)
            moves[count++] = move;
    }

    const Move* begin() const { return moves.data(); }
    const Move* end() const { return moves.data() + count; }
    Move* begin() { return moves.data(); }
    Move* end() { return moves.data() + count; }
    std::size_t size() const { return count; }
    bool empty() const { return count == 0; }
};

using FixedMoveList = MoveList<MAX_MOVES>;
struct TacticalMoveList : MoveList<MAX_TACTICAL_MOVES> {};

void generateAllMoves(const Position& position, bool whiteToMove, FixedMoveList& list);
void generateLegalMoves(Position& position, bool whiteToMove, FixedMoveList& list);
void generateCaptureMoves(Position& position, bool whiteToMove, FixedMoveList& list);

// Quiet checking moves for QSearch. These deliberately exclude captures and
// promotions because those are already supplied by generateTacticalMoves().
void generateQuietChecks(Position& position, bool whiteToMove, TacticalMoveList& list);

std::vector<Move> generateAllMoves(const Position& position, bool whiteToMove);
std::vector<Move> generateLegalMoves(Position& position, bool whiteToMove);
std::vector<Move> generateCaptureMoves(Position& position, bool whiteToMove);
TacticalMoveList generateTacticalMoves(const Position& position, bool whiteToMove);
bool isCheckmate(Position& position, bool whiteToMove);

#endif
