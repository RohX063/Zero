#ifndef ZERO_MOVEPICK_H
#define ZERO_MOVEPICK_H

#include <array>
#include <cstddef>

#include "history.h"
#include "move.h"
#include "movegen.h"

class Position;

class MovePicker {
public:
    MovePicker(const Position& position, const Move* moves, std::size_t count,
               Move ttMove = Move::none(), Move counterMove = Move::none(),
               Move killer1 = Move::none(), Move killer2 = Move::none(),
               const Zero::Search::HistoryTables* history = nullptr,
               Zero::Color side = Zero::WHITE,
               Piece previousPiece = EMPTY,
               Zero::Square previousTo = Zero::SQ_NONE);

    MovePicker(const Position& position, const FixedMoveList& moves,
               Move ttMove = Move::none(), Move counterMove = Move::none(),
               Move killer1 = Move::none(), Move killer2 = Move::none(),
               const Zero::Search::HistoryTables* history = nullptr,
               Zero::Color side = Zero::WHITE,
               Piece previousPiece = EMPTY,
               Zero::Square previousTo = Zero::SQ_NONE)
        : MovePicker(position, moves.begin(), moves.size(),
                     ttMove, counterMove, killer1, killer2,
                     history, side, previousPiece, previousTo) {}

    Move next_move();

public:
    struct ScoredMove {
        Move move{};
        int score = 0;
    };

    static constexpr std::size_t CAPACITY = MAX_MOVES;

private:
    const Position& position_;
    std::array<ScoredMove, CAPACITY> moves_{};
    std::size_t count_ = 0;
    std::size_t cursor_ = 0;
};

#endif
