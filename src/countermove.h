#ifndef ZERO_COUNTERMOVE_H
#define ZERO_COUNTERMOVE_H

#include <array>

#include "move.h"
#include "piece.h"
#include "types.h"

namespace Zero::Search {

// Countermove history maps the move that just happened to the move that
// most often proves useful as the reply in that resulting position..
class CounterMoveHistory {
public:
    static constexpr std::size_t PIECE_COUNT = PIECE_NB;
    static constexpr std::size_t SQUARE_COUNT = SQUARE_NB;

    CounterMoveHistory() { clear(); }

    void clear() {
        for (auto& row : table_)
            row.fill(Move::none());
    }

    Move probe(Piece previousPiece, Square previousTo) const {
        if (previousPiece >= PIECE_NB || previousTo >= SQUARE_NB)
            return Move::none();
        return table_[previousPiece][previousTo];
    }

    void update(Piece previousPiece, Square previousTo, Move reply) {
        if (previousPiece >= PIECE_NB || previousTo >= SQUARE_NB)
            return;
        table_[previousPiece][previousTo] = reply;
    }

private:
    std::array<std::array<Move, SQUARE_COUNT>, PIECE_COUNT> table_{};
};

} // namespace Zero::Search

#endif
