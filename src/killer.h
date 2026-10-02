#ifndef ZERO_KILLER_H
#define ZERO_KILLER_H

#include <array>
#include <cstddef>

#include "move.h"
#include "types.h"

namespace Zero::Search {

// Two killer moves are retained per search ply. A killer is a quiet move that
// caused a beta cutoff at the same ply in the current search.
class KillerMoves {
public:
    static constexpr std::size_t SLOTS = 2;
    static constexpr std::size_t PLIES = MAX_PLY + 2;

    KillerMoves() { clear(); }

    void clear() {
        for (auto& ply : table_)
            ply.fill(Move::none());
    }

    Move first(Depth ply) const {
        return ply >= 0 && static_cast<std::size_t>(ply) < PLIES
            ? table_[static_cast<std::size_t>(ply)][0]
            : Move::none();
    }

    Move second(Depth ply) const {
        return ply >= 0 && static_cast<std::size_t>(ply) < PLIES
            ? table_[static_cast<std::size_t>(ply)][1]
            : Move::none();
    }

    void update(Depth ply, Move move) {
        if (ply < 0 || static_cast<std::size_t>(ply) >= PLIES || !move.isOk())
            return;
        if (move == table_[static_cast<std::size_t>(ply)][0])
            return;

        auto& slots = table_[static_cast<std::size_t>(ply)];
        slots[1] = slots[0];
        slots[0] = move;
    }

private:
    std::array<std::array<Move, SLOTS>, PLIES> table_{};
};

} // namespace Zero::Search

#endif
