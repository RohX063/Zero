#include "search.h"

#include <algorithm>
#include <iostream>

#include "movegen.h"
#include "movepick.h"
#include "position.h"
#include "repetition.h"
#include "search_helpers.h"

namespace Zero::Search {

SearchResult Worker::searchRoot(Depth depth)
{
    resetNodes();
    deadline_ = {};
    clockNodes_ = 0;
    stopped_ = false;
    stats_ = {};
    tt_.new_search();
    killerMoves_.clear();
    rootBestMove_ = Move::none();
    rootDepth_ = depth;
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

    // The previous iteration's best move is always searched first. (The root
    // position is never stored in the TT by this loop, so the TT cannot be
    // relied on to carry the principal move between iterations.)
    const Move rootFirst = rootBestMove_.isOk()
        ? rootBestMove_
        : (rootTT.hit ? rootTT.move : Move::none());

    MovePicker picker(position_,
                      moves,
                      rootFirst,
                      Move::none(),
                      killerMoves_.first(0),
                      killerMoves_.second(0),
                      history_.get(),
                      us,
                      EMPTY,
                      SQ_NONE);

    Stack* ss = &stack_[0];
    *ss = Stack{};
    ss->ply = 0;

    bool firstMove = true;

    for (;;) {
        if (shouldStop())
            return result;      // result.completed stays false; bestMove may hold a usable partial

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

        if (Rules::moveCausesThreefold(position_, move)) {
            // Exact root-level repetition result: this move cannot score above
            // a draw, so do not let LMR/PVS/TT noise turn a forced repetition
            // into a candidate that looks better than an available win.
            score = VALUE_DRAW;
        } else if (firstMove) {
            score = -search(position_,
                            child,
                            std::max(0, depth - 1),
                            -VALUE_INFINITE,
                            VALUE_INFINITE,
                            false);
            firstMove = false;
        } else {
            score = -search(position_,
                            child,
                            std::max(0, depth - 1),
                            -result.score - 1,
                            -result.score,
                            true);

            if (score > result.score) {
                ++stats_.pvsReSearches;
                score = -search(position_,
                                child,
                                std::max(0, depth - 1),
                                -VALUE_INFINITE,
                                -result.score,
                                false);
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

    // A search with no depth and no clock would never end on its own; give it a
    // finite default instead of running to MAX_PLY.
    Depth depthCap = limits.depth;
    if (depthCap <= 0 && timeBudget_.maximumMs <= 0)
        depthCap = 16;

    SearchResult lastCompleted;
    for (Depth depth = 1; depth <= MAX_PLY - 1; ++depth) {
        if (depthCap > 0 && depth > depthCap)
            break;

        rootDepth_ = depth;
        SearchResult result = searchRootIteration(depth);

        if (!result.completed) {
            // Time ran out inside this iteration. The first root move is the
            // previous iteration's best move, searched at the new depth; if it
            // (or a move that beat it after a full re-search) finished, that
            // result is at least as trustworthy as the old one.
            if (result.bestMove.isOk() && lastCompleted.bestMove.isOk()) {
                lastCompleted.bestMove = result.bestMove;
                lastCompleted.score = result.score;
            } else if (result.bestMove.isOk() && !lastCompleted.bestMove.isOk()) {
                lastCompleted = result;
            }
            break;
        }

        lastCompleted = result;

        // Terminal root (checkmate/stalemate): the caller reports it; nothing to print or deepen.
        if (!result.bestMove.isOk())
            break;
        rootBestMove_ = result.bestMove;

        std::cout << "info depth " << depth
                  << " seldepth " << seldepth()
                  << " score " << uciScoreString(result.score)
                  << " time " << elapsedMs()
                  << " nodes " << nodes_
                  << " nps " << nps()
                  << " hashfull " << tt_.hashfull()
                  << " pv " << uciMoveString(result.bestMove)
                  << "\n" << std::flush;

        // A forced mate has been found: deeper iterations cannot improve it
        // for the side that is mating.
        if (limits.depth <= 0 && result.score >= VALUE_MATE - MAX_PLY
            && VALUE_MATE - result.score <= depth)
            break;

        const int elapsed = static_cast<int>(elapsedMs());
        stableIterations_ = updateStableIterations(
            lastBestMove_, result.bestMove, lastScore_, result.score, stableIterations_);

        const bool continueSearch = shouldContinueAfterIteration(
            timeBudget_,
            elapsed,
            lastBestMove_,
            result.bestMove,
            lastScore_,
            result.score,
            stableIterations_);

        lastBestMove_ = result.bestMove;
        lastScore_ = result.score;

        if (!continueSearch || shouldStop())
            break;
    }

    return lastCompleted;
}

void Worker::beginSearch(const Limits& limits)
{
    resetNodes();
    clockNodes_ = 0;
    stopped_ = false;
    stats_ = {};
    deadline_ = {};
    tt_.new_search();
    killerMoves_.clear();

    rootBestMove_ = Move::none();
    rootDepth_ = 0;

    searchStart_ = std::chrono::steady_clock::now();
    Limits adjusted = limits;
    adjusted.moveOverheadMs = params_.moveOverheadMs;
    timeBudget_ = computeTimeBudget(adjusted);
    lastBestMove_ = Move::none();
    lastScore_ = VALUE_DRAW;
    stableIterations_ = 0;

    if (timeBudget_.maximumMs > 0)
        deadline_ = searchStart_ + std::chrono::milliseconds(timeBudget_.maximumMs);
}

bool Worker::shouldStop() const
{
    // Sticky: once the deadline has been seen, every later call (including the
    // many calls made while the recursion unwinds) reports the same answer.
    if (stopped_)
        return true;

    if (deadline_ == std::chrono::steady_clock::time_point{})
        return false;

    if ((clockNodes_ & 1023ULL) != 0)
        return false;

    if (std::chrono::steady_clock::now() >= deadline_)
        stopped_ = true;

    return stopped_;
}

}
