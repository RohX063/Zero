#include "search.h"

#include <algorithm>
#include <iostream>

#include "movegen.h"
#include "movepick.h"
#include "position.h"
#include "repetition.h"

namespace Zero::Search {

SearchResult Worker::searchRoot(Depth depth)
{
    resetNodes();
    deadline_ = {};
    clockNodes_ = 0;
    stats_ = {};
    tt_.new_search();
    killerMoves_.clear();
    return searchRootIteration(depth);
}

SearchResult Worker::searchRootIteration(Depth depth)
{
    SearchResult result;

    if (Rules::isAutomaticDraw(position_)) {
        result.score = VALUE_DRAW;
        result.depth = depth;
        result.completed = true;
        return result;
    }

    FixedMoveList moves;
    generateLegalMoves(position_, position_.isWhiteToMove(), moves);
    if (moves.empty()) {
        result.depth = depth;
        result.completed = !shouldStop();
        return result;
    }

    const Color us = position_.isWhiteToMove() ? WHITE : BLACK;
    const TTData rootTT = tt_.probe(position_.key());

    MovePicker picker(position_,
                      moves,
                      rootTT.hit ? rootTT.move : Move::none(),
                      Move::none(),
                      killerMoves_.first(0),
                      killerMoves_.second(0),
                      &history_,
                      us,
                      EMPTY,
                      SQ_NONE);

    Stack* ss = &stack_[0];
    *ss = Stack{};
    ss->ply = 0;

    bool firstMove = true;

    for (;;) {
        if (shouldStop())
            return result;

        const Move move = picker.next_move();
        if (!move.isOk())
            break;

        const Piece attacker = position_.piece_on(move.from_sq());

        StateInfo newState;
        ss->currentMove = move;
        ss->movedPiece = attacker;

        position_.doMove(move, newState);

        Stack* child = ss + 1;
        *child = Stack{};
        child->ply = 1;
        child->currentMove = move;
        child->movedPiece = attacker;

        Value score;

        if (firstMove) {
            score = -search(position_,
                            child,
                            std::max(0, depth - 1),
                            -VALUE_INFINITE,
                            VALUE_INFINITE);
            firstMove = false;
        } else {
            score = -search(position_,
                            child,
                            std::max(0, depth - 1),
                            -result.score - 1,
                            -result.score);

            if (score > result.score) {
                ++stats_.pvsReSearches;
                score = -search(position_,
                                child,
                                std::max(0, depth - 1),
                                -VALUE_INFINITE,
                                -result.score);
            }
        }

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
    stats_ = {};
    deadline_ = {};
    tt_.new_search();
    killerMoves_.clear();

    const int budgetMs = computeTimeBudgetMs(limits);
    if (budgetMs > 0)
        deadline_ = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(budgetMs);
}

bool Worker::shouldStop() const
{
    if (deadline_ == std::chrono::steady_clock::time_point{})
        return false;

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

    const int reserve = std::min(250, std::max(10, limits.sideTimeMs / 50));
    return std::max(1, budget - reserve);
}

}
