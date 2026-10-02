#include "search.h"

#include <algorithm>
#include <iostream>

#include "movegen.h"
#include "movepick.h"
#include "position.h"

namespace Zero::Search {

SearchResult Worker::searchRoot(Depth depth)
{
    resetNodes();
    deadline_ = {};
    clockNodes_ = 0;
    tt_.new_search();
    killerMoves_.clear();
    return searchRootIteration(depth);
}

SearchResult Worker::searchRootIteration(Depth depth)
{
    FixedMoveList moves;
    generateLegalMoves(position_, position_.isWhiteToMove(), moves);
    if (moves.empty()) {
        SearchResult result;
        result.depth = depth;
        result.completed = !shouldStop();
        return result;
    }

    MovePicker picker(position_, moves);
    SearchResult result;
    Stack* ss = &stack_[0];
    *ss = Stack{};
    ss->ply = 0;

    for (;;) {
        if (shouldStop())
            return result;

        const Move move = picker.next_move();
        if (!move.isOk()) break;

        StateInfo newState;
        ss->currentMove = move;
        position_.doMove(move, newState);
        Stack* child = ss + 1;
        *child = Stack{};
        child->ply = 1;
        const Value score = -search(position_, child, depth - 1,
                                    -VALUE_INFINITE, VALUE_INFINITE);
        position_.undoMove(move);

        if (shouldStop())
            return result;

        if (score > result.score || !result.bestMove.isOk()) {
            result.score = score;
            result.bestMove = move;
        }
    }

    result.depth = depth;
    result.completed = true;
    return result;
}

SearchResult Worker::iterativeDeepening(Depth maxDepth)
{
    Limits limits;
    limits.depth = maxDepth;
    return iterativeDeepening(limits);
}

SearchResult Worker::iterativeDeepening(const Limits& limits)
{
    beginSearch(limits);

    SearchResult lastCompleted;
    for (Depth depth = 1; depth <= MAX_PLY - 1; ++depth) {
        if (limits.depth > 0 && depth > limits.depth)
            break;

        SearchResult result = searchRootIteration(depth);
        if (!result.completed)
            break;

        lastCompleted = result;
        std::cout << "info depth " << depth
                  << " score cp " << result.score
                  << " nodes " << nodes_
                  << " hashfull " << tt_.hashfull()
                  << std::endl;

        if (shouldStop())
            break;
    }

    return lastCompleted;
}

void Worker::beginSearch(const Limits& limits)
{
    resetNodes();
    clockNodes_ = 0;
    deadline_ = {};
    tt_.new_search();
    killerMoves_.clear();

    const int budgetMs = computeTimeBudgetMs(limits);
    if (budgetMs > 0)
        deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(budgetMs);
}

bool Worker::shouldStop() const
{
    if (deadline_ == std::chrono::steady_clock::time_point{})
        return false;

    // Amortize clock reads in the recursive hot path.
    if ((clockNodes_ & 1023ULL) != 0)
        return false;

    return std::chrono::steady_clock::now() >= deadline_;
}

int Worker::computeTimeBudgetMs(const Limits& limits) const
{
    if (limits.movetimeMs > 0)
        return limits.movetimeMs;

    if (limits.sideTimeMs <= 0)
        return 0;

    const int moves = limits.movesToGo > 0 ? limits.movesToGo : 30;
    int budget = limits.sideTimeMs / std::max(1, moves);
    budget += (3 * limits.incrementMs) / 4;

    // Preserve a small reserve for the GUI/clock boundary.
    const int reserve = std::min(250, std::max(10, limits.sideTimeMs / 50));
    return std::max(1, budget - reserve);
}

}
