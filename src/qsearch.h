#ifndef ZERO_QSEARCH_H
#define ZERO_QSEARCH_H

#include <array>
#include "search.h"
#include "movegen.h"

namespace Zero::Search {

class QMovePicker {
public:
    explicit QMovePicker(Position& position, bool whiteToMove, bool includeQuietChecks);
    Move next_move();

private:
    struct ScoredMove {
        Move move{};
        int score = 0;
    };

    static int scoreCapture(const Position& position, const Move& move);

    std::array<ScoredMove, MAX_TACTICAL_MOVES> moves_{};
    std::size_t count_ = 0;
    std::size_t cursor_ = 0;
};

class QSearch {
public:
    explicit QSearch(Worker& worker) : worker_(worker) {}
    // qply: how many plies deep inside the quiescence search this node is (0 = first qsearch node).
    Value run(Position& position, Stack* ss, Value alpha, Value beta, int qply = 0);
private:
    Worker& worker_;
};

}

#endif
